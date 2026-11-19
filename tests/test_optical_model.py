"""Tests for the optical model step and its raysect backend.

Most scenes are the unit square of ``square_mesh`` in the z=0 plane, seen from above by a
pinhole camera whose view lies completely inside the square. Expected values are computed
independently of irflow where possible (scipy quadrature of Planck's law, OpenCV projection,
raysect's tungsten data file read directly).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import ArrayLike
from scipy.constants import Boltzmann, Planck, Stefan_Boltzmann, speed_of_light
from scipy.integrate import quad

import irflow.io
from irflow import IRFlow, OpticalModel
from irflow.backends.optical.planck import apparent_temperature, planck_radiance
from irflow.data import CameraImages, HeatFluxField, SurfaceMesh, TemperatureField
from irflow.io.vtk import to_polydata

EMISSIVITY = 0.7
WAVELENGTH_RANGE = (3000.0, 5000.0)
SPECTRAL_BINS = 10


@pytest.fixture
def camera_config() -> dict[str, Any]:
    """Pinhole camera 1 m above the centre of the unit square, x to the right and y up.

    The surface is a grey body without reflections.
    """
    pytest.importorskip("raysect")
    pytest.importorskip("cv2")
    return {
        "camera": {
            "type": "pinhole",
            "position": [0.5, 0.5, 1.0],
            "look_at": [0.5, 0.5, 0.0],
            "up": [0.0, 1.0, 0.0],
            "fov": 40.0,  # sees x in 0.5 +- 0.36 m: only the square
            "pixels": [6, 4],
        },
        "materials": {"grey": {"type": "grey", "emissivity": EMISSIVITY, "reflection": "none"}},
        "surface_material": "grey",
        "wavelength_range": WAVELENGTH_RANGE,
        "pixel_samples": 2,
        "spectral_bins": SPECTRAL_BINS,
        "processes": 1,
    }


@pytest.fixture
def cv2() -> Any:
    """OpenCV, for the expected pixel positions."""
    return pytest.importorskip("cv2")


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


def rectangle(corner: ArrayLike, edge_1: ArrayLike, edge_2: ArrayLike) -> SurfaceMesh:
    """Parallelogram from ``corner`` spanned by ``edge_1`` and ``edge_2``, two triangles."""
    start, side_1, side_2 = np.asarray(corner), np.asarray(edge_1), np.asarray(edge_2)
    nodes = np.array([start, start + side_1, start + side_1 + side_2, start + side_2])
    return SurfaceMesh(nodes=nodes, triangles=np.array([[0, 1, 2], [0, 2, 3]]))


def cube_walls() -> tuple[SurfaceMesh, SurfaceMesh]:
    """The unit cube as its bottom face (z=0) and its other five faces: a closed enclosure."""
    nodes = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float)
    bottom = np.array([[0, 2, 6], [0, 6, 4]])
    others = [(1, 5, 7, 3), (0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6)]
    triangles = [[a, b, c] for a, b, c, _ in others] + [[a, c, d] for a, _, c, d in others]
    return SurfaceMesh(nodes, bottom), SurfaceMesh(nodes, np.array(triangles))


def render(config: dict[str, Any], field: TemperatureField, tmp_path: Path) -> CameraImages:
    optical = OpticalModel(config=config, input_data=field)
    optical.run(tmp_path)
    return optical.get_irflow_pass_data()


def pinhole_as_opencv(pinhole: dict[str, Any], cv2: Any) -> dict[str, Any]:
    """OpenCV camera config (no distortion) with the geometry of a ``pinhole`` camera config."""
    n_columns, n_rows = pinhole["pixels"]
    position = np.array(pinhole["position"], dtype=float)
    forward = np.array(pinhole["look_at"]) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, pinhole["up"])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.array([right, down, forward])  # rows: camera axes in world coordinates

    focal = max(n_columns, n_rows) / (2 * np.tan(np.radians(pinhole["fov"]) / 2))
    camera_matrix = [[focal, 0, (n_columns - 1) / 2], [0, focal, (n_rows - 1) / 2], [0, 0, 1]]
    return {
        "type": "opencv",
        "camera_matrix": camera_matrix,
        "distortion": [0.0] * 5,
        "r_vector": cv2.Rodrigues(rotation)[0].ravel().tolist(),
        "t_vector": (-rotation @ position).tolist(),
        "pixels": [n_columns, n_rows],
    }


def project(camera: dict[str, Any], point: np.ndarray, cv2: Any) -> np.ndarray:
    """OpenCV pixel coordinates (u, v) of a world point seen by an ``opencv`` camera config."""
    pixels, _ = cv2.projectPoints(
        np.asarray(point, dtype=float).reshape(-1, 3),
        np.array(camera["r_vector"]),
        np.array(camera["t_vector"]),
        np.array(camera["camera_matrix"]),
        np.array(camera["distortion"]),
    )
    return np.asarray(pixels).reshape(-1, 2).squeeze()


def centroid(image: np.ndarray) -> np.ndarray:
    """Radiance-weighted centre (u, v) = (column, row) of an image."""
    rows, columns = np.indices(image.shape)
    return np.array([(image * columns).sum(), (image * rows).sum()]) / image.sum()


# --- Planck's law and apparent temperature (no raysect needed) -------------------------------


def test_planck_integrates_to_stefan_boltzmann() -> None:
    temperature = 1000.0
    total = quad(planck_radiance, 100.0, 1e6, args=(temperature,), limit=200)[0]
    assert total == pytest.approx(Stefan_Boltzmann * temperature**4 / np.pi, rel=1e-3)


def test_apparent_temperature_inverts_band_radiance() -> None:
    delta = (WAVELENGTH_RANGE[1] - WAVELENGTH_RANGE[0]) / SPECTRAL_BINS
    wavelengths = WAVELENGTH_RANGE[0] + (np.arange(SPECTRAL_BINS) + 0.5) * delta
    temperature = np.array([60.0, 300.0, 1234.5, 9000.0, 5.0e4])
    radiance = [0.3 * planck_radiance(wavelengths, value).sum() * delta for value in temperature]

    result = apparent_temperature(np.array(radiance), wavelengths, delta, emissivity=0.3)
    np.testing.assert_allclose(result, temperature, rtol=1e-5)
    # nothing seen, colder than 50 K, hotter than 1e5 K
    outside = [0.0, 0.3 * planck_radiance(wavelengths, 40.0).sum() * delta, 1e12]
    assert np.all(np.isnan(apparent_temperature(np.array(outside), wavelengths, delta, 0.3)))


# --- Rendering an emitting surface -------------------------------------------------------------


def test_uniform_plane_gives_grey_body_radiance(
    tmp_path: Path, square_mesh: SurfaceMesh, camera_config: dict[str, Any]
) -> None:
    temperature = 600.0
    images = render(camera_config, uniform_field(square_mesh, temperature), tmp_path)

    assert images.radiance.shape == (1, 4, 6)
    assert images.wavelength_range == WAVELENGTH_RANGE
    np.testing.assert_allclose(images.radiance, EMISSIVITY * band_radiance(temperature), rtol=0.01)


@pytest.mark.parametrize(("emissivity", "reflection"), [(1.0, "none"), (0.6, "diffuse")])
def test_apparent_temperature_with_default_emissivity_is_true_temperature(
    tmp_path: Path,
    square_mesh: SurfaceMesh,
    camera_config: dict[str, Any],
    emissivity: float,
    reflection: str,
) -> None:
    # By default the assumed emissivity is the surface material's, so a grey plane (here open:
    # its reflection sees nothing) appears at its true temperature.
    camera_config["materials"]["grey"] = {
        "type": "grey",
        "emissivity": emissivity,
        "reflection": reflection,
    }
    images = render(camera_config, uniform_field(square_mesh, 777.0), tmp_path)

    assert images.assumed_emissivity == emissivity
    np.testing.assert_allclose(images.apparent_temperature, 777.0, rtol=1e-3)


def test_assumed_emissivity_can_be_overridden(
    tmp_path: Path, square_mesh: SurfaceMesh, camera_config: dict[str, Any]
) -> None:
    # Seen as a black body, a grey surface appears colder than it is
    camera_config["assumed_emissivity"] = 1.0
    images = render(camera_config, uniform_field(square_mesh, 777.0), tmp_path)

    assert images.assumed_emissivity == 1.0
    assert np.all(images.apparent_temperature < 760.0)


def test_one_image_per_time_and_hotter_is_brighter(
    tmp_path: Path, temperature_field: TemperatureField, camera_config: dict[str, Any]
) -> None:
    images = render(camera_config, temperature_field, tmp_path)

    np.testing.assert_allclose(images.times, temperature_field.times)
    assert images.radiance.shape == (2, 4, 6)
    assert images.apparent_temperature.shape == (2, 4, 6)
    assert np.all(images.radiance[1] > images.radiance[0])


def test_image_orientation(
    tmp_path: Path, square_mesh: SurfaceMesh, camera_config: dict[str, Any]
) -> None:
    # Hotter towards +x (image right) and towards +y (image top)
    x, y = square_mesh.nodes[:, 0], square_mesh.nodes[:, 1]
    temperature = np.array([400 + 200 * x + 100 * y])
    field = TemperatureField(mesh=square_mesh, times=np.array([0.0]), temperature=temperature)

    [image] = render(camera_config, field, tmp_path).radiance

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

    [image] = render(camera_config, field, tmp_path).radiance

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
    camera_config["camera"]["look_at"] = [0.5, 0.5, 2.0]
    images = render(camera_config, uniform_field(square_mesh, 600.0), tmp_path)
    assert np.all(images.radiance == 0.0)
    assert np.all(np.isnan(images.apparent_temperature))


# --- Materials: emission and reflection --------------------------------------------------------


def enclosure_images(
    tmp_path: Path, config: dict[str, Any], material: dict[str, Any], temperature: float
) -> CameraImages:
    """Camera inside the closed unit cube, all walls of ``material`` at ``temperature``.

    The bottom wall is the thermal model surface, the other five come from the ``scene``.
    """
    bottom, others = cube_walls()
    to_polydata(others).save(tmp_path / "walls.vtp")
    config["camera"] = {
        "type": "pinhole",
        "position": [0.5, 0.5, 0.6],
        "look_at": [0.4, 0.45, 0.0],
        "up": [0.0, 1.0, 0.0],
        "fov": 70.0,
        "pixels": [2, 2],
    }
    config["materials"] = {"wall": material}
    config["surface_material"] = "wall"
    walls = {"mesh": tmp_path / "walls.vtp", "material": "wall", "temperature": temperature}
    config["scene"] = [walls]
    return render(config, uniform_field(bottom, temperature), tmp_path)


@pytest.mark.parametrize(
    ("material", "max_reflections"),
    [
        ({"type": "tungsten"}, 300),  # reflectivity ~0.97: needs many reflections
        ({"type": "grey", "emissivity": 0.3, "reflection": "diffuse"}, 60),
        ({"type": "grey", "emissivity": 0.3, "reflection": "specular"}, 60),
    ],
)
def test_closed_isothermal_enclosure_looks_like_a_black_body(
    tmp_path: Path, camera_config: dict[str, Any], material: dict[str, Any], max_reflections: int
) -> None:
    # Kirchhoff: emissivity = 1 - reflectivity, so inside an isothermal enclosure emission plus
    # reflections add up to black body radiance, for any material.
    camera_config["max_reflections"] = max_reflections
    images = enclosure_images(tmp_path, camera_config, material, temperature=900.0)
    np.testing.assert_allclose(images.radiance, band_radiance(900.0), rtol=0.02)


@pytest.mark.parametrize("max_reflections", [0, 2])
def test_max_reflections_is_the_number_of_reflections(
    tmp_path: Path, camera_config: dict[str, Any], max_reflections: int
) -> None:
    # Grey mirror walls: emission e B, plus (1 - e)^n e B for the n-th reflection followed
    camera_config["max_reflections"] = max_reflections
    material = {"type": "grey", "emissivity": 0.4, "reflection": "specular"}
    images = enclosure_images(tmp_path, camera_config, material, temperature=900.0)

    expected = (1 - 0.6 ** (max_reflections + 1)) * band_radiance(900.0)
    np.testing.assert_allclose(images.radiance, expected, rtol=0.01)


def test_tungsten_normal_emissivity_matches_optical_constants(
    tmp_path: Path, camera_config: dict[str, Any]
) -> None:
    # Looking straight down on a tungsten plane: radiance = (1 - R(normal)) x Planck in the band
    camera_config["camera"].update({"fov": 4.0, "pixels": [2, 2]})
    camera_config["surface_material"] = "tungsten"
    camera_config["spectral_bins"] = 20
    plane = rectangle([-1.0, -1.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0])
    images = render(camera_config, uniform_field(plane, 1000.0), tmp_path)

    import raysect.optical.library.metal

    data_file = Path(raysect.optical.library.metal.__file__).parent / "data" / "tungsten.json"
    data = json.loads(data_file.read_text())
    wavelength = np.linspace(*WAVELENGTH_RANGE, 2001)
    n = np.interp(wavelength, data["wavelength"], data["index"])
    k = np.interp(wavelength, data["wavelength"], data["extinction"])
    emissivity = 1 - ((n - 1) ** 2 + k**2) / ((n + 1) ** 2 + k**2)
    expected = np.trapz(emissivity * planck_radiance(wavelength, 1000.0), wavelength)

    np.testing.assert_allclose(images.radiance, expected, rtol=0.005)
    assert 0.02 < expected / band_radiance(1000.0) < 0.04  # tungsten is a poor emitter in IR
    # The default assumed emissivity is the band mean of the normal emissivity; with it, the
    # apparent temperature is close to (not exactly) the true one, as emissivity varies in band.
    assert images.assumed_emissivity == pytest.approx(emissivity.mean(), rel=0.005)
    np.testing.assert_allclose(images.apparent_temperature, 1000.0, rtol=0.01)


def test_fresnel_reflectance_limits() -> None:
    pytest.importorskip("raysect")
    from irflow.backends.optical.raysect_materials import fresnel_reflectance

    metal = np.array([2.0 + 17.0j])
    normal = ((2.0 - 1) ** 2 + 17.0**2) / ((2.0 + 1) ** 2 + 17.0**2)
    assert fresnel_reflectance(metal, 1.0)[0] == pytest.approx(normal)
    assert fresnel_reflectance(metal, 0.0)[0] == pytest.approx(1.0)

    # Glass at Brewster's angle reflects no p-polarised light: R = R_s / 2
    glass = np.array([1.5 + 0.0j])
    cos_i = np.cos(np.arctan(1.5))
    cos_t = np.sqrt(1 - (1 - cos_i**2) / 1.5**2)
    r_s = (cos_i - 1.5 * cos_t) / (cos_i + 1.5 * cos_t)
    assert fresnel_reflectance(glass, cos_i)[0] == pytest.approx(0.5 * r_s**2)


@pytest.mark.parametrize("reflection", ["specular", "none"])
def test_hot_object_appears_in_mirror_where_geometry_says(
    tmp_path: Path, camera_config: dict[str, Any], cv2: Any, reflection: str
) -> None:
    # A cold mirror-like floor (z=0) and a hot black square at z=0.5 that the camera cannot
    # see directly. Its mirror image lies at z=-0.5, at the pixel OpenCV projects it to.
    hot_centre = np.array([1.5, 0.15, 0.5])
    hot_square = rectangle(hot_centre - [0.1, 0.1, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0])
    to_polydata(hot_square).save(tmp_path / "hot.vtp")
    pinhole = {
        "position": [0.0, 0.0, 1.0],
        "look_at": [1.5, 0.0, -0.5],
        "up": [0.0, 0.0, 1.0],
        "fov": 20.0,
        "pixels": [16, 12],
    }
    camera = pinhole_as_opencv(pinhole, cv2)
    camera_config["camera"] = camera
    camera_config["materials"] = {
        "mirror": {"type": "grey", "emissivity": 0.05, "reflection": reflection},
        "black": {"type": "grey", "emissivity": 1.0, "reflection": "none"},
    }
    camera_config["surface_material"] = "mirror"
    hot = {"mesh": tmp_path / "hot.vtp", "material": "black", "temperature": 1200.0}
    camera_config["scene"] = [hot]
    camera_config["pixel_samples"] = 10
    floor = rectangle([-3.0, -3.0, 0.0], [6.0, 0.0, 0.0], [0.0, 6.0, 0.0])
    [image] = render(camera_config, uniform_field(floor, 300.0), tmp_path).radiance

    reflected = 0.95 * band_radiance(1200.0)
    if reflection == "none":
        assert image.max() < 1e-3 * reflected  # not visible directly
        return
    spot = np.where(image > 0.01 * reflected, image, 0.0)
    expected = project(camera, hot_centre * [1, 1, -1], cv2)
    np.testing.assert_allclose(centroid(spot), expected, atol=0.3)
    assert 0.5 * reflected < image.max() <= 1.01 * reflected


# --- Cameras ----------------------------------------------------------------------------------


def distorted_camera(camera_config: dict[str, Any], cv2: Any) -> dict[str, Any]:
    """OpenCV version of the fixture's pinhole camera, with distortion and shifted centre."""
    camera = pinhole_as_opencv(camera_config["camera"], cv2)
    camera["distortion"] = [-0.1, 0.02, 0.001, -0.001, 0.0]
    camera["camera_matrix"][0][2] += 0.3
    camera["camera_matrix"][1][2] -= 0.2
    return camera


def test_opencv_camera_sees_point_where_opencv_projects_it(
    tmp_path: Path, camera_config: dict[str, Any], cv2: Any
) -> None:
    camera: dict[str, Any] = {
        "type": "opencv",
        "camera_matrix": [[45.0, 0.0, 12.7], [0.0, 44.0, 8.1], [0.0, 0.0, 1.0]],
        "distortion": [-0.3, 0.1, 0.002, -0.003, 0.0],
        "r_vector": [2.2, 0.3, -0.4],
        "t_vector": [0.1, 0.2, 2.5],
        "pixels": [24, 18],
    }
    rotation = cv2.Rodrigues(np.array(camera["r_vector"]))[0]
    # A world point at camera-frame position (0.15, -0.12, 1) x 2.5: right of and above centre
    point = rotation.T @ (np.array([0.15, -0.12, 1.0]) * 2.5 - camera["t_vector"])
    expected = project(camera, point, cv2)
    assert 17 < expected[0] < 21 and 1 < expected[1] < 6  # top right, as set up

    # A small hot square facing the camera (about 3 pixels wide), nothing else in the scene
    right, down = rotation[0] * 0.15, rotation[1] * 0.15
    patch = rectangle(point - (right + down) / 2, right, down)
    camera_config["camera"] = camera
    camera_config["pixel_samples"] = 40
    [image] = render(camera_config, uniform_field(patch, 1000.0), tmp_path).radiance

    np.testing.assert_allclose(centroid(image), expected, atol=0.15)


def gentle_gradient(mesh: SurfaceMesh) -> TemperatureField:
    """Temperature rising slowly to +x and +y: low sampling noise, but any flip shows."""
    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    temperature = np.array([500.0 + 10.0 * x + 5.0 * y])
    return TemperatureField(mesh=mesh, times=np.array([0.0]), temperature=temperature)


def test_undistorted_opencv_camera_equals_pinhole(
    tmp_path: Path, square_mesh: SurfaceMesh, camera_config: dict[str, Any], cv2: Any
) -> None:
    camera_config["pixel_samples"] = 50
    field = gentle_gradient(square_mesh)
    [pinhole_image] = render(camera_config, field, tmp_path).radiance

    camera_config["camera"] = pinhole_as_opencv(camera_config["camera"], cv2)
    [opencv_image] = render(camera_config, field, tmp_path).radiance

    np.testing.assert_allclose(opencv_image, pinhole_image, rtol=0.01)
    assert pinhole_image[0, -1] > 1.03 * pinhole_image[-1, 0]  # the gradient is visible


def test_vectors_camera_from_file_equals_opencv(
    tmp_path: Path, square_mesh: SurfaceMesh, camera_config: dict[str, Any], cv2: Any
) -> None:
    from irflow.backends.optical.raysect_cameras import opencv_lines_of_sight

    camera = distorted_camera(camera_config, cv2)
    origins, directions = opencv_lines_of_sight(camera)
    assert directions.shape == (4, 6, 3)  # (rows, columns, xyz)
    np.savez(tmp_path / "lines_of_sight.npz", pixel_origins=origins, pixel_directions=directions)

    camera_config["pixel_samples"] = 50
    field = gentle_gradient(square_mesh)
    camera_config["camera"] = camera
    [opencv_image] = render(camera_config, field, tmp_path).radiance
    camera_config["camera"] = {"type": "vectors", "file": tmp_path / "lines_of_sight.npz"}
    [vectors_image] = render(camera_config, field, tmp_path).radiance

    np.testing.assert_allclose(vectors_image, opencv_image, rtol=0.01)


def test_lines_of_sight_follow_opencv_projection(camera_config: dict[str, Any], cv2: Any) -> None:
    from irflow.backends.optical.raysect_cameras import opencv_lines_of_sight

    camera = distorted_camera(camera_config, cv2)
    origins, directions = opencv_lines_of_sight(camera)
    pixels = project(camera, origins + 1.3 * directions, cv2)

    rows, columns = np.indices((4, 6))
    np.testing.assert_allclose(pixels, np.column_stack([columns.ravel(), rows.ravel()]), atol=1e-6)


# --- Step behaviour and saving -----------------------------------------------------------------


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
    steps: list[Any] = [FixedTemperatureStep(temperature_field), optical]
    IRFlow(case_name="case", root_dir=tmp_path, steps=steps).run()

    assert optical.temperature is temperature_field
    images = irflow.io.load_images(tmp_path / "case" / "OpticalModel" / "images.pvd")
    result = optical.get_irflow_pass_data()
    np.testing.assert_allclose(images.radiance, result.radiance)
    np.testing.assert_allclose(images.apparent_temperature, result.apparent_temperature)
    assert images.assumed_emissivity == EMISSIVITY


# --- Config errors -----------------------------------------------------------------------------

DELETE = object()


def change(config: dict[str, Any], keys: tuple[str, ...], value: Any) -> None:
    """Set ``config[keys[0]][keys[1]]... = value``, or delete that key if value is DELETE."""
    for key in keys[:-1]:
        config = config[key]
    if value is DELETE:
        del config[keys[-1]]
    else:
        config[keys[-1]] = value


@pytest.mark.parametrize(
    ("keys", "value", "match"),
    [
        (("camera",), DELETE, "missing key 'camera'"),
        (("emissivity",), 0.5, "'emissivity' was replaced by a material"),
        (("camera_position",), [0.0, 0.0, 1.0], "replaced by the 'camera' dict"),
        (("wavelength_range",), [5000.0, 3000.0], "wavelength_range"),
        (("surface_material",), "copper", "unknown material 'copper'"),
        (("max_reflections",), -1, "max_reflections"),
        (("assumed_emissivity",), 0.0, "assumed_emissivity"),
        # pinhole camera
        (("camera", "type"), "fisheye", "camera 'type' must be one of"),
        (("camera", "fov"), 180.0, "fov"),
        (("camera", "up"), [0.0, 0.0, 1.0], "parallel"),
        (("camera", "look_at"), [0.5, 0.5, 1.0], "differ"),
        (("camera", "pixels"), [64], "pixels"),
        (("camera", "focus"), 1.0, "unknown key 'focus'"),
        # materials
        (("materials", "grey", "type"), "plastic", "'type' must be one of"),
        (("materials", "grey", "emissivity"), 1.5, "emissivity must be in"),
        (("materials", "grey", "reflection"), "glossy", "reflection must be one of"),
        (("materials", "grey", "reflection"), DELETE, "missing key 'reflection'"),
        (("materials", "metal"), {"type": "tungsten", "roughness": 0.1}, "unknown key"),
        # scene
        (("scene",), [{"mesh": "none.vtp", "material": "grey", "temperature": 3e2}], "not found"),
        (("scene",), [{"mesh": "none.vtp", "material": "grey"}], "missing key 'temperature'"),
    ],
)
def test_invalid_config(
    camera_config: dict[str, Any], keys: tuple[str, ...], value: Any, match: str
) -> None:
    change(camera_config, keys, value)
    with pytest.raises(ValueError, match=match):
        OpticalModel(config=camera_config)


def test_tungsten_needs_band_inside_its_optical_data(camera_config: dict[str, Any]) -> None:
    camera_config["surface_material"] = "tungsten"
    camera_config["wavelength_range"] = [7000.0, 14000.0]
    with pytest.raises(ValueError, match="tungsten optical data cover"):
        OpticalModel(config=camera_config)


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("camera_matrix", [[1.0, 0.0], [0.0, 1.0]], "camera_matrix must be 3x3"),
        ("camera_matrix", [[-9.0, 0, 3.0], [0, 9.0, 2.0], [0, 0, 1]], "focal lengths"),
        ("distortion", [0.1, 0.0, 0.0, 0.0], "distortion"),
        ("distortion", [-3.0, 0.0, 0.0, 0.0, 0.0], "cannot be inverted"),
        ("r_vector", [0.0, 0.0], "r_vector"),
        ("t_vector", DELETE, "missing key 't_vector'"),
    ],
)
def test_invalid_opencv_camera(
    camera_config: dict[str, Any], cv2: Any, key: str, value: Any, match: str
) -> None:
    camera_config["camera"] = pinhole_as_opencv(camera_config["camera"], cv2)
    change(camera_config, ("camera", key), value)
    with pytest.raises(ValueError, match=match):
        OpticalModel(config=camera_config)


@pytest.mark.parametrize(
    ("arrays", "match"),
    [
        (None, "not found"),
        ({"pixel_origins": np.zeros((4, 6, 3))}, "no array"),
        ({"pixel_origins": np.zeros((4, 6, 3)), "pixel_directions": np.ones((6, 4, 3))}, "shape"),
        ({"pixel_origins": np.zeros((4, 6, 3)), "pixel_directions": np.zeros((4, 6, 3))}, "zero"),
    ],
)
def test_invalid_vectors_camera(
    tmp_path: Path, camera_config: dict[str, Any], arrays: dict[str, Any] | None, match: str
) -> None:
    path = tmp_path / "lines_of_sight.npz"
    if arrays is not None:
        np.savez(path, **arrays)
    camera_config["camera"] = {"type": "vectors", "file": path}
    with pytest.raises(ValueError, match=match):
        OpticalModel(config=camera_config)


def test_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unknown optical model backend"):
        OpticalModel(backend_code="nope")
