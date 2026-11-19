"""Tests for saving and loading step results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import irflow.io
from irflow.data import HeatFluxField, TemperatureField


def test_heat_flux_round_trip(tmp_path: Path, heat_flux_field: HeatFluxField) -> None:
    [path] = irflow.io.save(heat_flux_field, tmp_path, ["vtk"])
    assert path == tmp_path / "heat_flux.vtp"

    loaded = irflow.io.load_heat_flux(path)
    np.testing.assert_allclose(loaded.mesh.nodes, heat_flux_field.mesh.nodes)
    np.testing.assert_array_equal(loaded.mesh.triangles, heat_flux_field.mesh.triangles)
    np.testing.assert_allclose(loaded.heat_flux, heat_flux_field.heat_flux)


def test_temperature_round_trip(tmp_path: Path, temperature_field: TemperatureField) -> None:
    [path] = irflow.io.save(temperature_field, tmp_path, ["VTK"])
    assert path == tmp_path / "temperature.pvd"
    assert len(list((tmp_path / "temperature").glob("*.vtp"))) == 2

    loaded = irflow.io.load_temperature(path)
    np.testing.assert_allclose(loaded.times, temperature_field.times)
    np.testing.assert_allclose(loaded.temperature, temperature_field.temperature)
    np.testing.assert_array_equal(loaded.mesh.triangles, temperature_field.mesh.triangles)


def test_load_heat_flux_from_other_code(gyrohit_target_vtk: Path) -> None:
    field = irflow.io.load_heat_flux(gyrohit_target_vtk, array_name="Heat-flux[W/m^2]")
    np.testing.assert_allclose(field.heat_flux, [1.0e6, 2.0e6])

    with pytest.raises(KeyError, match="Heat-flux"):
        irflow.io.load_heat_flux(gyrohit_target_vtk)


def test_unknown_file_types(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        irflow.io.load_heat_flux(tmp_path / "flux.csv")
    with pytest.raises(ValueError):
        irflow.io.load_temperature(tmp_path / "temperature.vtp")


def test_check_formats() -> None:
    assert irflow.io.check_formats(["VTK"]) == ("vtk",)
    assert irflow.io.check_formats([]) == ()
    with pytest.raises(ValueError, match="hdf5"):
        irflow.io.check_formats(["vtk", "hdf5"])
    with pytest.raises(TypeError):
        irflow.io.check_formats("vtk")
