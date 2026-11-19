"""End-to-end run through IRFlow with backends that need no external codes.

Gyrohit reads a precomputed heat flux, simple1d turns it into surface temperature, IRFlow
passes the data along and saves both results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import irflow.io
from irflow import FieldLineTracing, IRFlow, ThermalModel

MATERIAL = {
    "density": {"type": "constant", "value": 3060},
    "heat_capacity": {"type": "constant", "value": 1000},
    "heat_conductivity": {"type": "constant", "value": 5},
}


def run_chain(root_dir: Path, gyrohit_target_vtk: Path, time_stepping: dict) -> IRFlow:
    flt = FieldLineTracing(
        backend_code="gyrohit",
        config={"target_mesh_path": str(gyrohit_target_vtk), "target_mesh_type": "vtk"},
    )
    tm = ThermalModel(
        backend_code="simple1d",
        config={
            "thickness": 0.01,
            "material_properties": MATERIAL,
            "back_boundary": {"type": "temperature", "value": 373.0},
            "initial_temperature": 373.0,
            "time_stepping": time_stepping,
        },
    )
    flow = IRFlow(case_name="chain", root_dir=root_dir, steps=[flt, tm])
    flow.run()
    return flow


def test_steady_chain_gives_analytic_node_temperatures(
    tmp_path: Path, gyrohit_target_vtk: Path
) -> None:
    flow = run_chain(tmp_path, gyrohit_target_vtk, {"type": "steady"})

    # Triangles get T_back + q L / k = 373 + [2000, 4000] K; nodes 0 and 2 are shared by both
    # (equal areas), node 1 belongs to the first triangle only, node 3 to the second only.
    loaded = irflow.io.load_temperature(flow.case_dir / "ThermalModel" / "temperature.pvd")
    np.testing.assert_allclose(loaded.times, [0.0])
    np.testing.assert_allclose(loaded.temperature[0], [3373.0, 2373.0, 3373.0, 4373.0])

    heat_flux = irflow.io.load_heat_flux(flow.case_dir / "FieldLineTracing" / "heat_flux.vtp")
    np.testing.assert_allclose(heat_flux.heat_flux, [1.0e6, 2.0e6])


def test_transient_chain_saves_every_time(tmp_path: Path, gyrohit_target_vtk: Path) -> None:
    time_stepping = {"type": "transient", "time_step": 0.01, "end_time": 0.05}
    flow = run_chain(tmp_path, gyrohit_target_vtk, time_stepping)

    loaded = irflow.io.load_temperature(flow.case_dir / "ThermalModel" / "temperature.pvd")
    np.testing.assert_allclose(loaded.times, [0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
    np.testing.assert_allclose(loaded.temperature[0], 373.0)
    assert np.all(np.diff(loaded.temperature, axis=0) > 0)  # heated surface warms up
    # the hotter triangle's own node (3) is hotter than the cooler triangle's (1)
    assert np.all(loaded.temperature[1:, 3] > loaded.temperature[1:, 1])
