"""Tests for saving and loading step results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

import irflow.io
from irflow.data import CameraImages, HeatFluxField, TemperatureField


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


def test_images_round_trip(tmp_path: Path, camera_images: CameraImages) -> None:
    [path] = irflow.io.save(camera_images, tmp_path, ["vtk"])
    assert path == tmp_path / "images.pvd"
    assert len(list((tmp_path / "images").glob("*.vti"))) == 2

    loaded = irflow.io.load_images(path)
    np.testing.assert_allclose(loaded.times, camera_images.times)
    np.testing.assert_allclose(loaded.radiance, camera_images.radiance)
    assert loaded.wavelength_range == camera_images.wavelength_range
    np.testing.assert_allclose(loaded.apparent_temperature, camera_images.apparent_temperature)
    assert loaded.assumed_emissivity == camera_images.assumed_emissivity


def test_images_are_upright_in_vtk(tmp_path: Path, camera_images: CameraImages) -> None:
    # VTK images start at the bottom row, so the first VTK cells are the last image row
    irflow.io.save(camera_images, tmp_path, ["vtk"])
    grid = pv.read(tmp_path / "images" / "image_0000.vti")
    assert grid.dimensions == (5, 4, 1)
    np.testing.assert_allclose(grid.cell_data["radiance"][:4], camera_images.radiance[0, -1])
    np.testing.assert_allclose(
        grid.cell_data["apparent_temperature"][-4:], camera_images.apparent_temperature[0, 0]
    )


def test_save_rejects_unknown_data(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        irflow.io.save(np.zeros(3), tmp_path, ["vtk"])  # type: ignore[arg-type]


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
    with pytest.raises(ValueError):
        irflow.io.load_images(tmp_path / "image_0000.vti")


def test_check_formats() -> None:
    assert irflow.io.check_formats(["VTK"]) == ("vtk",)
    assert irflow.io.check_formats([]) == ()
    with pytest.raises(ValueError, match="hdf5"):
        irflow.io.check_formats(["vtk", "hdf5"])
    with pytest.raises(TypeError):
        irflow.io.check_formats("vtk")
