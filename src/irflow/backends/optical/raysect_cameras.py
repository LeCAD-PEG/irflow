"""Cameras for the raysect optical backend.

The ``camera`` config is a dict whose ``type`` selects the model:

- ``"pinhole"``: ideal pinhole from position, viewing direction and field of view
  (:class:`RadianceCamera`).
- ``"opencv"``: a calibrated camera in the OpenCV model (camera matrix, distortion, pose), as
  from ``cv2.calibrateCamera`` / ``cv2.solvePnP`` or Calcam.
- ``"vectors"``: any calibration exported as one line of sight per pixel (``.npz`` file).

Conventions, the same for all types: lengths in m in the world frame of the meshes. Images are
indexed [row, column] with row 0 at the image top and column 0 at the left, as an image from
the real camera. OpenCV pixel coordinates (u, v) are (column, row) with integer values at pixel
centres, so pixel (column c, row r) covers c - 0.5 < u < c + 0.5 and r - 0.5 < v < r + 0.5.
Every camera ray has weight 1, so pixels report radiance with no cosine fall-off towards the
image edges, as a radiometrically calibrated camera.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from raysect.core import Point3D, Vector3D, rotate_basis, translate
from raysect.core.math.random import uniform
from raysect.optical import Ray, World
from raysect.optical.observer import FullFrameSampler2D, PinholeCamera, RadiancePipeline2D
from raysect.optical.observer.base import Observer2D

from irflow.config import check_config

# Config keys per camera type: (required, optional), besides "type"
CAMERA_KEYS: dict[str, tuple[list[str], list[str]]] = {
    "pinhole": (["position", "look_at", "fov", "pixels"], ["up"]),
    "opencv": (["camera_matrix", "distortion", "r_vector", "t_vector", "pixels"], []),
    "vectors": (["file"], []),
}

UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)
MAX_UNDISTORT_ERROR = 0.01  # [pixel], reprojection error of the inverted distortion


class RadianceCamera(PinholeCamera):
    """Raysect pinhole camera whose pixels report the radiance of the scene.

    Raysect's ``PinholeCamera`` weights every ray with the cosine of its angle to the optical
    axis, which darkens the image towards its edges. A radiometrically calibrated IR camera has
    no such fall-off, so here every ray has weight 1.
    """

    def _generate_rays(
        self, x: int, y: int, template: Ray, ray_count: int
    ) -> list[tuple[Ray, float]]:
        rays = super()._generate_rays(x, y, template, ray_count)
        return [(ray, 1.0) for ray, _ in rays]


class LineOfSightCamera(Observer2D):
    """Camera given by a line of sight per pixel; rays fill exactly the pixel's footprint.

    ``origins`` (ny, nx, 3) are the start points of the lines of sight [m] and
    ``corner_directions`` (ny + 1, nx + 1, 3) the directions through the pixel corners: pixel
    (row r, column c) lies between corners [r, c] (top left) and [r + 1, c + 1] (bottom right).
    Each ray goes through a uniformly random point of the pixel: a bilinear blend of the four
    corner directions. Every ray has weight 1, so the pixel value is the mean radiance.

    Raysect's own ``VectorCamera`` is not used: it blends between the neighbouring pixels'
    vectors, so each pixel samples an area two pixels wide (and edge pixels a single direction).
    """

    def __init__(
        self,
        origins: np.ndarray,
        corner_directions: np.ndarray,
        pipelines: list[RadiancePipeline2D],
        parent: World,
    ) -> None:
        n_rows, n_columns = origins.shape[:2]
        if corner_directions.shape != (n_rows + 1, n_columns + 1, 3):
            raise ValueError(
                f"corner_directions must have shape {(n_rows + 1, n_columns + 1, 3)}, "
                f"got {corner_directions.shape}"
            )
        super().__init__((n_columns, n_rows), FullFrameSampler2D(), pipelines, parent=parent)
        self.origins = origins
        self.corner_directions = corner_directions

    def _generate_rays(
        self, x: int, y: int, template: Ray, ray_count: int
    ) -> list[tuple[Ray, float]]:
        # raysect pixel x is the image column, y the image row
        corners = self.corner_directions
        # raysect's random generator is seeded separately in every render process
        across = np.array([uniform() for _ in range(ray_count)])[:, None]
        down = np.array([uniform() for _ in range(ray_count)])[:, None]
        directions = (
            (1 - across) * (1 - down) * corners[y, x]
            + across * (1 - down) * corners[y, x + 1]
            + (1 - across) * down * corners[y + 1, x]
            + across * down * corners[y + 1, x + 1]
        )
        origin = Point3D(*self.origins[y, x])
        return [
            (template.copy(origin, Vector3D(*direction).normalise()), 1.0)
            for direction in directions
        ]

    def _pixel_sensitivity(self, x: int, y: int) -> float:
        return 1.0


def check_camera(camera: Mapping[str, Any]) -> None:
    """Raise ValueError if the ``camera`` config cannot be used, before any long run starts."""
    if not isinstance(camera, Mapping):
        raise ValueError(f"camera must be a dict with a 'type', got {camera!r}")
    kind = camera.get("type")
    if kind not in CAMERA_KEYS:
        raise ValueError(f"camera 'type' must be one of {sorted(CAMERA_KEYS)}, got {kind!r}")
    required, optional = CAMERA_KEYS[kind]
    check_config(camera, ["type", *required], optional, where=f"OpticalModel camera ({kind!r})")

    # Building the geometry checks all values; it is cheap.
    if kind == "pinhole":
        pinhole_transform(camera)
        check_pixels(camera["pixels"])
        fov = float(camera["fov"])
        if not 0.0 < fov < 180.0:
            raise ValueError(f"camera fov must be between 0 and 180 degrees, got {fov}")
    elif kind == "opencv":
        opencv_corner_directions(camera)
    else:
        read_lines_of_sight(camera["file"])


def make_camera(
    camera: Mapping[str, Any], pipeline: RadiancePipeline2D, world: World
) -> Observer2D:
    """Raysect observer for the ``camera`` config, placed in ``world``."""
    kind = camera["type"]
    if kind == "pinhole":
        return RadianceCamera(
            pixels=check_pixels(camera["pixels"]),
            fov=float(camera["fov"]),
            pipelines=[pipeline],
            frame_sampler=FullFrameSampler2D(),
            parent=world,
            transform=pinhole_transform(camera),
        )
    if kind == "opencv":
        origin, corners = opencv_corner_directions(camera)
        n_rows, n_columns = corners.shape[0] - 1, corners.shape[1] - 1
        origins = np.broadcast_to(origin, (n_rows, n_columns, 3))
        return LineOfSightCamera(origins, corners, [pipeline], world)

    origins, directions = read_lines_of_sight(camera["file"])
    return LineOfSightCamera(origins, corners_from_centres(directions), [pipeline], world)


def as_vector(value: Any, length: int, name: str) -> np.ndarray:
    """Check that ``value`` is ``length`` finite numbers and return them as an array."""
    vector = np.asarray(value, dtype=float)
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"camera {name} must be {length} finite numbers, got {value}")
    return vector


def check_pixels(pixels: Sequence[int]) -> tuple[int, int]:
    """Image size (nx, ny): columns, rows, each at least 2."""
    if len(pixels) != 2 or min(pixels) < 2:
        raise ValueError(f"camera pixels must be (nx, ny), each at least 2, got {pixels}")
    return int(pixels[0]), int(pixels[1])


def pinhole_transform(camera: Mapping[str, Any]) -> Any:
    """Raysect transform of a pinhole camera at ``position`` looking at ``look_at``."""
    position = as_vector(camera["position"], 3, "position")
    view = as_vector(camera["look_at"], 3, "look_at") - position
    up = as_vector(camera.get("up", (0.0, 0.0, 1.0)), 3, "up")
    if not np.any(view):
        raise ValueError("camera look_at must differ from position")
    if not np.any(up):
        raise ValueError("camera up must not be zero")
    cross = np.cross(view / np.linalg.norm(view), up / np.linalg.norm(up))
    if np.linalg.norm(cross) < 1e-6:
        raise ValueError("camera up must not be parallel to the viewing direction")
    return translate(*position) * rotate_basis(Vector3D(*view), Vector3D(*up))


def opencv_corner_directions(camera: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Pinhole position (3,) and directions through the pixel corners (ny + 1, nx + 1, 3)."""
    n_columns, n_rows = check_pixels(camera["pixels"])
    # Pixel centres are at integer pixel coordinates, so the corners are half-way between.
    u, v = np.meshgrid(np.arange(n_columns + 1) - 0.5, np.arange(n_rows + 1) - 0.5)
    return opencv_directions(camera, u, v)


def opencv_lines_of_sight(camera: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre lines of sight of an ``opencv`` camera config, e.g. for a ``vectors`` file.

    Returns ``pixel_origins`` and ``pixel_directions``, each (ny, nx, 3), row 0 = image top.
    """
    n_columns, n_rows = check_pixels(camera["pixels"])
    u, v = np.meshgrid(np.arange(n_columns, dtype=float), np.arange(n_rows, dtype=float))
    origin, directions = opencv_directions(camera, u, v)
    return np.broadcast_to(origin, directions.shape).copy(), directions


def opencv_directions(
    camera: Mapping[str, Any], u: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pinhole position (3,) and unit world directions (..., 3) through pixel coordinates (u, v).

    OpenCV model: a world point X is at x_cam = R X + t in the camera frame (x right, y down,
    z along the optical axis), with R = Rodrigues(r_vector) and t = t_vector. So the pinhole is
    at -R^T t, and a camera-frame direction d points along R^T d in the world.
    """
    camera_matrix = np.asarray(camera["camera_matrix"], dtype=float)
    if camera_matrix.shape != (3, 3) or not np.all(np.isfinite(camera_matrix)):
        raise ValueError(f"camera camera_matrix must be 3x3 finite numbers, got {camera_matrix}")
    if camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0:
        raise ValueError("camera camera_matrix focal lengths fx, fy must be positive")
    if not np.allclose(camera_matrix[2], (0.0, 0.0, 1.0)):
        raise ValueError("camera camera_matrix last row must be (0, 0, 1)")
    distortion = as_vector(camera["distortion"], 5, "distortion (k1, k2, p1, p2, k3)")
    rotation = cv2.Rodrigues(as_vector(camera["r_vector"], 3, "r_vector"))[0]
    t_vector = as_vector(camera["t_vector"], 3, "t_vector")

    pixel_points = np.stack([u, v], axis=-1).reshape(-1, 1, 2)
    # cv2's type stubs do not allow None for R and P, which cv2 accepts (no rectification)
    normalised = cv2.undistortPointsIter(  # type: ignore[call-overload]
        pixel_points, camera_matrix, distortion, None, None, UNDISTORT_CRITERIA
    ).reshape(-1, 2)
    camera_frame = np.column_stack([normalised, np.ones(len(normalised))])

    # The iterative inversion of the distortion can fail for strong distortion: check it.
    reprojected = cv2.projectPoints(
        camera_frame, np.zeros(3), np.zeros(3), camera_matrix, distortion
    )[0].reshape(-1, 2)
    error = np.abs(reprojected - pixel_points.reshape(-1, 2)).max()
    if not error < MAX_UNDISTORT_ERROR:
        raise ValueError(
            f"camera distortion cannot be inverted over the whole image (pixel error {error:.3g})"
        )

    directions = camera_frame @ rotation  # row vectors: (R^T d)^T = d^T R
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return -rotation.T @ t_vector, directions.reshape(*np.shape(u), 3)


def read_lines_of_sight(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read ``pixel_origins`` and ``pixel_directions`` (ny, nx, 3) from an ``.npz`` file.

    Row 0 is the image top, column 0 its left side. Directions are normalised here.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"camera file not found: {path}")
    with np.load(path) as data:
        missing = {"pixel_origins", "pixel_directions"} - set(data.files)
        if missing:
            raise ValueError(f"camera file {path} has no array(s) {sorted(missing)}")
        origins = np.asarray(data["pixel_origins"], dtype=float)
        directions = np.asarray(data["pixel_directions"], dtype=float)

    if directions.ndim != 3 or directions.shape[2] != 3 or min(directions.shape[:2]) < 2:
        raise ValueError(
            f"pixel_directions in {path} must have shape (ny, nx, 3) with nx, ny >= 2, "
            f"got {directions.shape}"
        )
    if origins.shape != directions.shape:
        raise ValueError(
            f"pixel_origins {origins.shape} and pixel_directions {directions.shape} in {path} "
            "must have the same shape"
        )
    length = np.linalg.norm(directions, axis=2)
    if not (np.all(np.isfinite(origins)) and np.all(np.isfinite(length)) and length.min() > 0):
        raise ValueError(f"pixel origins and directions in {path} must be finite and non-zero")
    return origins, directions / length[:, :, None]


def corners_from_centres(directions: np.ndarray) -> np.ndarray:
    """Directions through pixel corners (ny + 1, nx + 1, 3) from those through pixel centres.

    A corner direction is the mean of its four neighbouring pixel directions; outside the image
    the pixel directions are extrapolated linearly.
    """
    padded = np.pad(directions, ((1, 1), (1, 1), (0, 0)), mode="reflect", reflect_type="odd")
    corners = padded[:-1, :-1] + padded[1:, :-1] + padded[:-1, 1:] + padded[1:, 1:]
    return np.asarray(corners / np.linalg.norm(corners, axis=2)[:, :, None])
