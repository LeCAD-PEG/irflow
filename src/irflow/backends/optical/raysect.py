"""Optical model backend for raysect: a pinhole IR camera looking at the hot surface.

The surface becomes a raysect ``Mesh``. Its material, :class:`TemperatureEmitter`, emits
grey-body radiation: emissivity x Planck spectral radiance at the local temperature. For every
ray that hits the mesh, raysect passes the material a ``MeshIntersection`` holding the index of
the triangle that was hit and the barycentric coordinates (u, v, w) of the hit point, so the
temperature is interpolated linearly from the triangle's three node values.

Each pixel value is the radiance [W/(m^2 sr)] integrated over the camera band and averaged
over the rays of the pixel. Reflections are not modelled yet (see :func:`surface_material`).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from raysect.core import (
    AffineMatrix3D,
    Intersection,
    Normal3D,
    Point3D,
    Primitive,
    Vector3D,
    rotate_basis,
    translate,
)
from raysect.core.workflow import MulticoreEngine, SerialEngine
from raysect.optical import Ray, Spectrum, World
from raysect.optical.material import Material, NullVolume
from raysect.optical.observer import FullFrameSampler2D, PinholeCamera, RadiancePipeline2D
from raysect.primitive import Mesh

from irflow.backends.base import OpticalModelBackend
from irflow.backends.optical.planck import planck_radiance
from irflow.data import CameraImages, SurfaceMesh, TemperatureField


class TemperatureEmitter(NullVolume):
    """Raysect material: grey-body emission of a triangle mesh with temperatures at its nodes.

    ``triangles`` (M, 3) must be the triangles of the raysect ``Mesh`` in the same order, so
    that the triangle index raysect reports for a hit points into it. Set ``node_temperature``
    (N,) [K] before rendering. Both faces of a triangle emit.
    """

    def __init__(self, triangles: np.ndarray, emissivity: float) -> None:
        super().__init__()
        self.triangles = triangles
        self.emissivity = emissivity
        self.node_temperature = np.zeros(0)

    def evaluate_surface(
        self,
        world: World,
        ray: Ray,
        primitive: Primitive,
        hit_point: Point3D,
        exiting: bool,
        inside_point: Point3D,
        outside_point: Point3D,
        normal: Normal3D,
        world_to_primitive: AffineMatrix3D,
        primitive_to_world: AffineMatrix3D,
        intersection: Intersection,
    ) -> Spectrum:
        """Spectral radiance [W/(m^2 sr nm)] leaving the surface at the hit point."""
        temperature = self.temperature_at(intersection)
        spectrum = ray.new_spectrum()
        spectrum.samples[:] = self.emissivity * planck_radiance(spectrum.wavelengths, temperature)
        return spectrum

    def temperature_at(self, intersection: Intersection) -> float:
        """Temperature [K] at a hit point: barycentric interpolation of the node values."""
        nodes = self.triangles[intersection.triangle]
        weights = (intersection.u, intersection.v, intersection.w)
        return float(np.dot(weights, self.node_temperature[nodes]))


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


def surface_material(emitter: TemperatureEmitter) -> Material:
    """Material of the surface mesh: for now only its own emission.

    To add reflections, combine the emitter with a reflecting material, e.g. for an opaque grey
    surface ``raysect.optical.material.Add(Lambert(ConstantSF(1 - emissivity)), emitter)``;
    other surfaces of the machine would then have to be in the scene as well.
    """
    return emitter


def non_degenerate_triangles(mesh: SurfaceMesh) -> np.ndarray:
    """Triangles of ``mesh`` with non-zero area, as raysect sees them (float32 coordinates).

    Raysect would otherwise drop degenerate triangles itself and renumber the rest, so that its
    triangle indices would no longer match ours. Dropping them changes nothing: they have no
    area and cannot be hit.
    """
    corners = mesh.nodes.astype(np.float32).astype(float)[mesh.triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    return mesh.triangles[np.linalg.norm(normals, axis=1) > 0.0]


def as_point(value: Sequence[float], name: str) -> np.ndarray:
    """Check that ``value`` is a finite 3D vector and return it as an array."""
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be three finite numbers (x, y, z), got {value}")
    return point


class BackendRaysect(OpticalModelBackend):
    """Connector between the irflow optical model step and raysect.

    The camera is an ideal pinhole camera. Config keys (keyword arguments):

    - ``camera_position``: (x, y, z) of the pinhole [m]
    - ``look_at``: (x, y, z) of the point seen at the image centre [m]
    - ``fov``: full field of view across the longer image side [deg]
    - ``pixels``: (nx, ny) image size: columns, rows
    - ``emissivity``: constant grey emissivity of the surface, 0 < emissivity <= 1
    - ``up``: direction that points up in the image (default (0, 0, 1)); must not be parallel
      to the viewing direction
    - ``wavelength_range``: (min, max) camera band [nm] (default (3000, 5000))
    - ``pixel_samples``: rays per pixel, spread over the pixel and averaged (default 10)
    - ``spectral_bins``: wavelength bins over the band (default 20)
    - ``processes``: CPU processes for raysect (default None: all CPUs; 1: no subprocesses)
    """

    def __init__(
        self,
        *,
        camera_position: Sequence[float],
        look_at: Sequence[float],
        fov: float,
        pixels: Sequence[int],
        emissivity: float,
        up: Sequence[float] = (0.0, 0.0, 1.0),
        wavelength_range: Sequence[float] = (3000.0, 5000.0),
        pixel_samples: int = 10,
        spectral_bins: int = 20,
        processes: int | None = None,
    ) -> None:
        self.camera_position = as_point(camera_position, "camera_position")
        self.look_at = as_point(look_at, "look_at")
        self.up = as_point(up, "up")
        self.fov = float(fov)
        if len(pixels) != 2 or len(wavelength_range) != 2:
            raise ValueError("pixels and wavelength_range must have two values each")
        self.pixels = (int(pixels[0]), int(pixels[1]))
        self.emissivity = float(emissivity)
        self.wavelength_range = (float(wavelength_range[0]), float(wavelength_range[1]))
        self.pixel_samples = int(pixel_samples)
        self.spectral_bins = int(spectral_bins)
        self.processes = processes
        self.check_config()
        self.result: CameraImages | None = None

    def check_config(self) -> None:
        """Raise ValueError for a config raysect cannot use, before any long run starts."""
        view = self.look_at - self.camera_position
        if not np.any(view):
            raise ValueError("look_at must differ from camera_position")
        if not np.any(self.up):
            raise ValueError("up must not be zero")
        cross = np.cross(view / np.linalg.norm(view), self.up / np.linalg.norm(self.up))
        if np.linalg.norm(cross) < 1e-6:
            raise ValueError("up must not be parallel to the viewing direction")
        if not 0.0 < self.fov < 180.0:
            raise ValueError(f"fov must be between 0 and 180 degrees, got {self.fov}")
        if min(self.pixels) < 2:
            raise ValueError(f"pixels must be (nx, ny), each at least 2, got {self.pixels}")
        if not 0.0 < self.emissivity <= 1.0:
            raise ValueError(f"emissivity must be in (0, 1], got {self.emissivity}")
        if not 0.0 < self.wavelength_range[0] < self.wavelength_range[1]:
            raise ValueError(
                f"wavelength_range must be (min, max) with 0 < min < max [nm], "
                f"got {self.wavelength_range}"
            )
        if self.pixel_samples < 1 or self.spectral_bins < 1:
            raise ValueError("pixel_samples and spectral_bins must be at least 1")
        if self.processes is not None and self.processes < 1:
            raise ValueError(f"processes must be None or at least 1, got {self.processes}")

    def run(self, step_dir: Path, temperature: TemperatureField) -> None:
        """Render one camera image per time of ``temperature``."""
        if not np.all(np.isfinite(temperature.temperature)) or temperature.temperature.min() <= 0:
            raise ValueError("Raysect: surface temperatures must be finite and above 0 K")

        world = World()
        triangles = non_degenerate_triangles(temperature.mesh)
        emitter = TemperatureEmitter(triangles, self.emissivity)
        # Our triangles are already free of degenerate ones (tolerant=False keeps the numbering);
        # closed=False because the plasma-facing surface is in general not a closed volume.
        Mesh(
            vertices=temperature.mesh.nodes,
            triangles=triangles,
            smoothing=False,
            closed=False,
            tolerant=False,
            parent=world,
            material=surface_material(emitter),
        )
        camera, pipeline = self.make_camera(world)

        images = []
        for index, time in enumerate(temperature.times):
            print(f"Raysect: rendering image {index + 1}/{len(temperature.times)} (t = {time} s)")
            emitter.node_temperature = temperature.temperature[index]
            camera.observe()
            # raysect frames are indexed [x, y] with x = 0 left and y = 0 top
            images.append(pipeline.frame.mean.T.copy())

        self.result = CameraImages(
            times=temperature.times,
            radiance=np.array(images),
            wavelength_range=self.wavelength_range,
        )

    def make_camera(self, world: World) -> tuple[RadianceCamera, RadiancePipeline2D]:
        """Pinhole camera at ``camera_position`` looking at ``look_at``, and its radiance output."""
        pipeline = RadiancePipeline2D(display_progress=False, accumulate=False)
        forward = Vector3D(*(self.look_at - self.camera_position))
        camera = RadianceCamera(
            pixels=self.pixels,
            fov=self.fov,
            pipelines=[pipeline],
            frame_sampler=FullFrameSampler2D(),
            parent=world,
            transform=translate(*self.camera_position) * rotate_basis(forward, Vector3D(*self.up)),
        )
        camera.pixel_samples = self.pixel_samples
        camera.spectral_bins = self.spectral_bins
        # Set the maximum first: raysect's default band (375-740 nm) lies below the IR band.
        camera.max_wavelength = self.wavelength_range[1]
        camera.min_wavelength = self.wavelength_range[0]
        if self.processes == 1:
            camera.render_engine = SerialEngine()
        else:
            camera.render_engine = MulticoreEngine(processes=self.processes)
        camera.quiet = True
        return camera, pipeline

    def get_result(self) -> CameraImages:
        """One radiance image per time [W/(m^2 sr)]."""
        if self.result is None:
            raise RuntimeError("Raysect backend has not been run yet.")
        return self.result
