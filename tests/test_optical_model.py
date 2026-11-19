"""Tests for the optical model step and its raysect backend.

The scene is the unit square of ``square_mesh`` in the z=0 plane, seen from above by a camera
whose view lies completely inside the square.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.constants import Boltzmann, Planck, Stefan_Boltzmann, speed_of_light
from scipy.integrate import quad

import irflow.io
from irflow import IRFlow, OpticalModel
from irflow.backends.optical.planck import planck_radiance
from irflow.data import HeatFluxField, SurfaceMesh, TemperatureField

EMISSIVITY = 0.7
WAVELENGTH_RANGE = (3000.0, 5000.0)


@pytest.fixture
def camera_config() -> dict[str, Any]:
    """Raysect camera 1 m above the centre of the unit square, x to the right and y up."""
    pytest.importorskip("raysect")
    return {
        "camera_position": [0.5, 0.5, 1.0],
        "look_at": [0.5, 0.5, 0.0],
        "up": [0.0, 1.0, 0.0],
        "fov": 40.0,  # sees x in 0.5 +- 0.36 m: only the square
        "pixels": [6, 4],
        "emissivity": EMISSIVITY,
        "wavelength_range": WAVELENGTH_RANGE,
        "pixel_samples": 2,
        "spectral_bins": 10,
        "processes": 1,
    }


def band_radiance(temperature: float) -> float:
    """Black body radiance in WAVELENGTH_RANGE [W/(m^2 sr)], computed independently of irflow."""

    def planck(wavelength_nm: float) -> float:
        wavelength = wavelength_nm * 1e-9
        exponent = Planck * speed_of_light / (wavelength * Boltzmann * temperature)
        per_metre = 2 * Planck * speed_of_light**2 / wavelength**5 / np.expm1(exponent)
        return per_metre * 1e-9

    return quad(planck, *WAVELENGTH_RANGE)[0]


def uniform_field(mesh: SurfaceMesh, temperature: float) -> TemperatureField:
    return TemperatureField(
        mesh=mesh, times=np.array([0.0]), temperature=np.full((1, mesh.n_nodes), temperature)
    )


def test_planck_integrates_to_stefan_boltzmann() -> None:
    temperature = 1000.0
    total = quad(planck_radiance, 100.0, 1e6, args=(temperature,), limit=200)[0]
    assert total == pytest.approx(Stefan_Boltzmann * temperature**4 / np.pi, rel=1e-3)


def test_uniform_plane_gives_grey_body_radiance(
    tmp_path: Path, square_mesh: SurfaceMesh, camera_config: dict[str, Any]
) -> None:
    temperature = 600.0
    field = uniform_field(square_mesh, temperature)
    optical = OpticalModel(config=camera_config, input_data=field)
    optical.run(tmp_path)
    images = optical.get_irflow_pass_data()

    assert images.radiance.shape == (1, 4, 6)
    assert images.wavelength_range == WAVELENGTH_RANGE
    expected = EMISSIVITY * band_radiance(temperature)
    np.testing.assert_allclose(images.radiance, expected, rtol=0.01)


def test_one_image_per_time_and_hotter_is_brighter(
    tmp_path: Path, temperature_field: TemperatureField, camera_config: dict[str, Any]
) -> None:
    optical = OpticalModel(config=camera_config, input_data=temperature_field)
    optical.run(tmp_path)
    images = optical.get_irflow_pass_data()

    np.testing.assert_allclose(images.times, temperature_field.times)
    assert images.radiance.shape == (2, 4, 6)
    assert np.all(images.radiance[1] > images.radiance[0])


def test_image_orientation(
    tmp_path: Path, square_mesh: SurfaceMesh, camera_config: dict[str, Any]
) -> None:
    # Hotter towards +x (image right) and towards +y (image top)
    x, y = square_mesh.nodes[:, 0], square_mesh.nodes[:, 1]
    temperature = np.array([400 + 200 * x + 100 * y])
    field = TemperatureField(mesh=square_mesh, times=np.array([0.0]), temperature=temperature)

    optical = OpticalModel(config=camera_config, input_data=field)
    optical.run(tmp_path)
    [image] = optical.get_irflow_pass_data().radiance

    assert image[:, :3].mean() < image[:, 3:].mean()
    assert image[:2, :].mean() > image[2:, :].mean()


def test_temperature_is_interpolated_inside_triangles(
    tmp_path: Path, square_mesh: SurfaceMesh, camera_config: dict[str, Any]
) -> None:
    # T = 500 + 100 x is linear, so barycentric interpolation reproduces it exactly anywhere on
    # the triangles; a wrong vertex order would give wrong temperatures inside them.
    camera_config["pixel_samples"] = 50
    temperature = np.array([500.0 + 100.0 * square_mesh.nodes[:, 0]])
    field = TemperatureField(mesh=square_mesh, times=np.array([0.0]), temperature=temperature)

    optical = OpticalModel(config=camera_config, input_data=field)
    optical.run(tmp_path)
    [image] = optical.get_irflow_pass_data().radiance

    # The 6 columns split the view, x in 0.5 +- tan(20 deg), into equal strips (left = low x)
    half_width = np.tan(np.radians(20.0))
    edges = np.linspace(0.5 - half_width, 0.5 + half_width, 7)
    for column in range(6):
        x0, x1 = edges[column], edges[column + 1]
        mean_band = quad(lambda x: band_radiance(500.0 + 100.0 * x), x0, x1)[0] / (x1 - x0)
        assert image[:, column].mean() == pytest.approx(EMISSIVITY * mean_band, rel=0.02)


def test_camera_looking_away_sees_nothing(
    tmp_path: Path, square_mesh: SurfaceMesh, camera_config: dict[str, Any]
) -> None:
    camera_config["look_at"] = [0.5, 0.5, 2.0]
    optical = OpticalModel(config=camera_config, input_data=uniform_field(square_mesh, 600.0))
    optical.run(tmp_path)
    assert np.all(optical.get_irflow_pass_data().radiance == 0.0)


def test_input_data_file(
    tmp_path: Path, temperature_field: TemperatureField, camera_config: dict[str, Any]
) -> None:
    [path] = irflow.io.save(temperature_field, tmp_path, ["vtk"])
    optical = OpticalModel(config=camera_config, input_data=path)
    assert optical.has_own_input
    assert optical.temperature is not None
    np.testing.assert_allclose(optical.temperature.temperature, temperature_field.temperature)


def test_needs_temperature_before_run(tmp_path: Path, camera_config: dict[str, Any]) -> None:
    optical = OpticalModel(config=camera_config)
    assert not optical.has_own_input
    with pytest.raises(RuntimeError, match="no surface temperature"):
        optical.run(tmp_path)
    with pytest.raises(RuntimeError, match="no result yet"):
        optical.get_irflow_pass_data()


def test_rejects_heat_flux_as_input(
    camera_config: dict[str, Any], heat_flux_field: HeatFluxField
) -> None:
    optical = OpticalModel(config=camera_config)
    with pytest.raises(TypeError, match="TemperatureField"):
        optical.set_irflow_pass_data(heat_flux_field)


class FixedTemperatureStep:
    """Stand-in for a thermal model step that returns a given temperature field."""

    name = "fixed_temperature"
    enabled = True
    has_own_input = True
    dir_name = "ThermalModel"

    def __init__(self, field: TemperatureField) -> None:
        self.field = field

    def run(self, step_dir: Path) -> None:
        pass

    def get_irflow_pass_data(self) -> TemperatureField:
        return self.field

    def set_irflow_pass_data(self, data: Any) -> None:
        raise AssertionError("the first step takes no data")


def test_irflow_passes_temperature_and_saves_images(
    tmp_path: Path, temperature_field: TemperatureField, camera_config: dict[str, Any]
) -> None:
    optical = OpticalModel(config=camera_config)
    steps = [FixedTemperatureStep(temperature_field), optical]
    IRFlow(case_name="case", root_dir=tmp_path, steps=steps).run()

    assert optical.temperature is temperature_field
    images = irflow.io.load_images(tmp_path / "case" / "OpticalModel" / "images.pvd")
    np.testing.assert_allclose(images.radiance, optical.get_irflow_pass_data().radiance)


@pytest.mark.parametrize(
    ("key", "value", "error"),
    [
        ("emisivity", 0.5, ValueError),  # misspelled key
        ("emissivity", 1.5, ValueError),
        ("fov", 180.0, ValueError),
        ("up", [0.0, 0.0, 1.0], ValueError),  # parallel to the viewing direction
        ("look_at", [0.5, 0.5, 1.0], ValueError),  # same as the camera position
        ("wavelength_range", [5000.0, 3000.0], ValueError),
        ("pixels", [64], ValueError),
    ],
)
def test_invalid_config(
    camera_config: dict[str, Any], key: str, value: Any, error: type[Exception]
) -> None:
    camera_config[key] = value
    with pytest.raises(error):
        OpticalModel(config=camera_config)


def test_missing_config_key(camera_config: dict[str, Any]) -> None:
    del camera_config["camera_position"]
    with pytest.raises(ValueError, match="missing key 'camera_position'"):
        OpticalModel(config=camera_config)


def test_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unknown optical model backend"):
        OpticalModel(backend_code="nope")
