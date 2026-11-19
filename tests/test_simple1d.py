"""Tests for the simple1d thermal model backend (independent 1D slabs) against analytic results."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from irflow import ThermalModel
from irflow.backends.tm.simple1d import (
    BackBoundary,
    BackendSimple1D,
    Material,
    TransientSlabs,
    cell_to_node_average,
    slab_grid,
)
from irflow.data import HeatFluxField, SurfaceMesh

RHO, C, K = 3060.0, 1000.0, 5.0  # SiO2-like density, heat capacity, heat conductivity
SIO2 = Material(density=RHO, heat_capacity=C, heat_conductivity=K)


def constant(value: float) -> dict[str, Any]:
    return {"type": "constant", "value": value}


@pytest.fixture
def config() -> dict[str, Any]:
    """Steady simple1d config with a fixed back temperature."""
    return {
        "thickness": 0.01,
        "material_properties": {
            "density": constant(RHO),
            "heat_capacity": constant(C),
            "heat_conductivity": constant(K),
        },
        "back_boundary": {"type": "temperature", "value": 373.0},
        "time_stepping": {"type": "steady"},
    }


def one_triangle(heat_flux: float) -> HeatFluxField:
    """A single triangle: every node then carries that triangle's surface temperature."""
    mesh = SurfaceMesh(nodes=np.eye(3), triangles=np.array([[0, 1, 2]]))
    return HeatFluxField(mesh=mesh, heat_flux=np.array([heat_flux]))


def surface_temperature(config: dict[str, Any], heat_flux: float) -> tuple[np.ndarray, np.ndarray]:
    """Run the backend on one triangle; return times and the surface temperature per time."""
    backend = BackendSimple1D(config)
    backend.run(Path("unused"), one_triangle(heat_flux))
    result = backend.get_result()
    assert np.allclose(result.temperature, result.temperature[:, :1])  # same at every node
    return result.times, result.temperature[:, 0]


def test_steady_fixed_back_temperature(config: dict[str, Any]) -> None:
    q = 2.0e6
    times, temperature = surface_temperature(config, q)
    assert list(times) == [0.0]
    assert temperature[0] == pytest.approx(373.0 + q * 0.01 / K)


def test_steady_htc(config: dict[str, Any]) -> None:
    q, h = 2.0e6, 1.0e4
    config["back_boundary"] = {"type": "htc", "value": h, "T_ext": 300.0}
    _, temperature = surface_temperature(config, q)
    assert temperature[0] == pytest.approx(300.0 + q / h + q * 0.01 / K)


@pytest.mark.parametrize(
    "grid", [{"n_points": 401}, {"n_points": 51, "first_cell_size": 5e-6}], ids=["fine", "graded"]
)
def test_transient_early_times_match_semi_infinite_solid(
    config: dict[str, Any], grid: dict[str, Any]
) -> None:
    q, t0, dt = 1.0e7, 373.0, 1e-3
    config.update(grid)
    config["initial_temperature"] = t0
    config["time_stepping"] = {"type": "transient", "time_step": dt, "end_time": 0.1}

    times, temperature = surface_temperature(config, q)

    assert times == pytest.approx(dt * np.arange(101))
    assert temperature[0] == t0
    # The heat reaches ~0.8 mm by the end, far less than the 10 mm slab: semi-infinite solid.
    # Backward Euler needs ~10 steps before its time error in the rise drops to ~1.5 %.
    exact_rise = 2 * q * np.sqrt(times / (np.pi * K * RHO * C))
    late = times >= 10 * dt
    assert temperature[late] - t0 == pytest.approx(exact_rise[late], rel=0.02)


def test_adiabatic_back_conserves_energy() -> None:
    thickness, t0, dt, n_steps = 0.002, 300.0, 0.01, 50
    heat_flux = np.array([0.0, 1.0e5, 3.0e6])
    solver = TransientSlabs(slab_grid(thickness, 21), SIO2, BackBoundary("adiabatic"), dt)

    temperature = np.full((len(heat_flux), 21), t0)
    for _ in range(n_steps):
        temperature = solver.step(temperature, heat_flux)

    mean_temperature = temperature @ solver.capacity / (RHO * C * thickness)
    expected = t0 + heat_flux * n_steps * dt / (RHO * C * thickness)
    assert mean_temperature == pytest.approx(expected, rel=1e-12)


def test_adiabatic_back_late_surface_temperature(config: dict[str, Any]) -> None:
    """Once the heat has crossed the slab, T_s = T0 + q t / (rho c L) + q L / (3 k)."""
    q, t0, thickness, end_time = 1.0e6, 300.0, 0.001, 1.0
    config["thickness"] = thickness
    config["initial_temperature"] = t0
    config["back_boundary"] = {"type": "adiabatic"}
    config["time_stepping"] = {"type": "transient", "time_step": 0.01, "end_time": end_time}

    _, temperature = surface_temperature(config, q)

    expected = t0 + q * end_time / (RHO * C * thickness) + q * thickness / (3 * K)
    assert temperature[-1] == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize(
    "back_boundary",
    [{"type": "adiabatic"}, {"type": "temperature", "value": 350.0}],
    ids=["adiabatic", "temperature"],
)
def test_zero_heat_flux_keeps_initial_temperature(
    config: dict[str, Any], back_boundary: dict[str, Any]
) -> None:
    config["back_boundary"] = back_boundary
    config["initial_temperature"] = 350.0
    config["time_stepping"] = {"type": "transient", "time_step": 0.01, "end_time": 0.1}

    _, temperature = surface_temperature(config, 0.0)
    assert temperature == pytest.approx(np.full(11, 350.0), rel=1e-12)


def test_node_values_are_area_weighted_averages() -> None:
    # Triangle 0 has area 0.5 and triangle 1 area 1.5; they share nodes 1 and 2.
    # Node 4 belongs to no triangle.
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [3, 1, 0], [5, 5, 5]], dtype=float)
    mesh = SurfaceMesh(nodes=nodes, triangles=np.array([[0, 1, 2], [1, 3, 2]]))
    cell_values = np.array([[10.0, 20.0], [0.0, 4.0]])  # two times

    node_values = cell_to_node_average(mesh, cell_values)

    shared = [(0.5 * 10.0 + 1.5 * 20.0) / 2.0, 1.5 * 4.0 / 2.0]
    assert node_values[:, :4] == pytest.approx(
        np.array([[10.0, shared[0], shared[0], 20.0], [0.0, shared[1], shared[1], 4.0]])
    )
    assert np.isnan(node_values[:, 4]).all()


def test_runs_as_thermal_model_step(
    tmp_path: Path, config: dict[str, Any], heat_flux_field: HeatFluxField
) -> None:
    config["time_stepping"] = {"type": "transient", "time_step": 0.1, "end_time": 0.5}
    config["initial_temperature"] = 373.0
    config["first_cell_size"] = 1e-5
    tm = ThermalModel(backend_code="simple1d", config=config, input_data=heat_flux_field)

    tm.run(tmp_path)
    result = tm.get_irflow_pass_data()

    assert result.mesh is heat_flux_field.mesh
    assert result.temperature.shape == (6, heat_flux_field.mesh.n_nodes)
    assert result.temperature[0] == pytest.approx(373.0)
    assert np.all(np.diff(result.temperature, axis=0) > 0)  # heating everywhere
    assert list(tmp_path.iterdir()) == []  # irflow, not the backend, saves the result


def test_graded_grid() -> None:
    nodes_x = slab_grid(0.01, 51, first_cell_size=5e-6)
    cell_sizes = np.diff(nodes_x)
    assert len(nodes_x) == 51
    assert nodes_x[0] == 0.0 and nodes_x[-1] == 0.01
    assert cell_sizes[0] == pytest.approx(5e-6)
    assert np.all(np.diff(cell_sizes) > 0)


def test_warns_when_surface_cell_is_too_coarse(config: dict[str, Any]) -> None:
    config["time_stepping"] = {"type": "transient", "time_step": 1e-3, "end_time": 0.1}
    with pytest.warns(UserWarning, match="first_cell_size"):
        BackendSimple1D(config)

    config["first_cell_size"] = 5e-6
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        BackendSimple1D(config)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"thickness": None}, "missing the key 'thickness'"),
        ({"material_properties": None}, "missing the key 'material_properties'"),
        ({"back_boundary": None}, "missing the key 'back_boundary'"),
        ({"thickness": -0.01}, "'thickness' must be a positive number"),
        ({"thickness": "1cm"}, "'thickness' must be a positive number"),
        ({"n_points": 2}, "'n_points' must be at least 3"),
        ({"n_points": 10.5}, "'n_points' must be an integer"),
        ({"first_cell_size": 0.001}, "first_cell_size .* must not be larger"),
        ({"back_boundary": {"type": "radiation"}}, "back_boundary type must be"),
        ({"back_boundary": {"type": "htc", "value": 1e4}}, "missing the key 'T_ext'"),
        ({"back_boundary": {"type": "temperature"}}, "missing the key 'value'"),
        ({"back_boundary": {"type": "adiabatic"}}, "adiabatic back has no solution"),
        ({"time_stepping": {"type": "implicit"}}, "type must be 'steady' or 'transient'"),
        ({"time_stepping": {"type": "transient", "time_step": 0.1}}, "missing the key 'end_time'"),
        (
            {"time_stepping": {"type": "transient", "time_step": 0.3, "end_time": 1.0}},
            "must be a multiple of time_step",
        ),
        ({"material_properties": {"density": constant(RHO)}}, "missing the key 'heat_capacity'"),
    ],
)
def test_config_errors(config: dict[str, Any], change: dict[str, Any], message: str) -> None:
    for key, value in change.items():
        if value is None:
            del config[key]
        else:
            config[key] = value
    with pytest.raises(ValueError, match=message):
        BackendSimple1D(config)


def test_only_constant_material_properties(config: dict[str, Any]) -> None:
    config["material_properties"]["heat_conductivity"] = {
        "type": "polynomial",
        "value": [21.8, -0.014],
    }
    with pytest.raises(NotImplementedError, match="heat_conductivity"):
        BackendSimple1D(config)
