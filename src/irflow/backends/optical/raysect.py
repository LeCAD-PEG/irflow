"""Optical model backend for raysect: a calibrated IR camera looking at the hot surface.

The thermal model surface becomes a raysect ``Mesh`` whose material emits at the local surface
temperature (interpolated from the node temperatures inside each triangle) and reflects what
it sees, see :mod:`irflow.backends.optical.raysect_materials`. Further machine components can
be added with the ``scene`` config; they occlude, emit (at a constant temperature) and reflect.
Rays that leave the scene contribute nothing (cold, black surroundings). Cameras are described
in :mod:`irflow.backends.optical.raysect_cameras`.

Each pixel value is the radiance [W/(m^2 sr)] summed over the spectral bins of the camera band
(flat spectral response) and averaged over the rays of the pixel. The apparent temperature of
each pixel inverts that radiance for a grey body of emissivity ``assumed_emissivity``, with the
same spectral bins (:func:`irflow.backends.optical.planck.apparent_temperature`).
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from raysect.core.workflow import MulticoreEngine, SerialEngine
from raysect.optical import Spectrum, World
from raysect.optical.material import Material
from raysect.optical.observer import RadiancePipeline2D
from raysect.optical.observer.base import Observer2D
from raysect.primitive import Mesh

from irflow.backends.base import OpticalModelBackend
from irflow.backends.optical.planck import apparent_temperature
from irflow.backends.optical.raysect_cameras import check_camera, make_camera
from irflow.backends.optical.raysect_materials import (
    MATERIAL_LIBRARY,
    ConstantTemperature,
    NodeTemperature,
    band_normal_emissivity,
    check_material,
    make_material,
)
from irflow.config import check_config
from irflow.data import CameraImages, SurfaceMesh, TemperatureField
from irflow.io.vtk import read_triangle_mesh


def non_degenerate_triangles(mesh: SurfaceMesh) -> np.ndarray:
    """Triangles of ``mesh`` with non-zero area, as raysect sees them (float32 coordinates).

    Raysect would otherwise drop degenerate triangles itself and renumber the rest, so that its
    triangle indices would no longer match ours. Dropping them changes nothing: they have no
    area and cannot be hit.
    """
    corners = mesh.nodes.astype(np.float32).astype(float)[mesh.triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    return mesh.triangles[np.linalg.norm(normals, axis=1) > 0.0]


def add_mesh(world: World, mesh: SurfaceMesh, triangles: np.ndarray, material: Material) -> None:
    """Add the ``triangles`` (already free of degenerate ones) of ``mesh`` to ``world``."""
    # tolerant=False keeps raysect's triangle numbering equal to ours; closed=False because a
    # surface mesh is in general not a closed volume.
    Mesh(
        vertices=mesh.nodes,
        triangles=triangles,
        smoothing=False,
        closed=False,
        tolerant=False,
        parent=world,
        material=material,
    )


class BackendRaysect(OpticalModelBackend):
    """Connector between the irflow optical model step and raysect.

    Config keys (keyword arguments):

    - ``camera``: camera model, a dict with ``type`` "pinhole", "opencv" or "vectors", see
      :mod:`irflow.backends.optical.raysect_cameras`
    - ``wavelength_range``: (min, max) camera band [nm], flat response (default (3000, 5000))
    - ``surface_material``: material name of the thermal model surface (default "tungsten")
    - ``materials``: dict name -> material spec, merged over the built-in library (which has
      "tungsten"), see :mod:`irflow.backends.optical.raysect_materials`
    - ``scene``: list of further machine components, each a dict with ``mesh`` (triangular
      surface VTK file [m], world frame), ``material`` (name) and ``temperature`` (constant [K]).
      They must not overlap the thermal model surface (cut that part out of them).
    - ``max_reflections``: reflections followed per ray (default 5); every path is followed up
      to this depth (no random termination). Inside a closed reflective enclosure, the missing
      radiance is about reflectivity^(max_reflections + 1), e.g. 0.97^6 for tungsten.
    - ``assumed_emissivity``: emissivity used to convert radiance to apparent temperature,
      0 < value <= 1. Default: the emissivity of ``surface_material``, as an IR camera analysis
      would assume it: a grey material's ``emissivity``; for tungsten, whose emissivity depends
      on wavelength and angle, its normal-incidence emissivity averaged over the spectral bins
      of the camera band (flat response). A tungsten surface at T then appears at about T, not
      exactly: the radiance weights the bins differently, and oblique views and reflections
      change the radiance. Set 1 for the black body "radiance temperature".
    - ``pixel_samples``: rays per pixel, spread over the pixel and averaged (default 10).
      Mirror reflections need no more; diffuse reflections are sampled with one random ray per
      hit, so a hot object covering a fraction p of the hemisphere seen by a diffuse surface
      gives a relative noise of about 1 / sqrt(p x pixel_samples) in its reflection.
    - ``spectral_bins``: wavelength bins over the band (default 20)
    - ``processes``: CPU processes for raysect (default None: all CPUs; 1: no subprocesses)
    """

    def __init__(
        self,
        *,
        camera: Mapping[str, Any],
        wavelength_range: Sequence[float] = (3000.0, 5000.0),
        surface_material: str = "tungsten",
        materials: Mapping[str, Mapping[str, Any]] | None = None,
        scene: Sequence[Mapping[str, Any]] | None = None,
        max_reflections: int = 5,
        assumed_emissivity: float | None = None,
        pixel_samples: int = 10,
        spectral_bins: int = 20,
        processes: int | None = None,
    ) -> None:
        self.camera = camera
        if len(wavelength_range) != 2:
            raise ValueError(f"wavelength_range must be (min, max) [nm], got {wavelength_range}")
        self.wavelength_range = (float(wavelength_range[0]), float(wavelength_range[1]))
        self.surface_material = surface_material
        self.materials = {**MATERIAL_LIBRARY, **(materials or {})}
        self.scene = list(scene or [])
        self.max_reflections = int(max_reflections)
        self.pixel_samples = int(pixel_samples)
        self.spectral_bins = int(spectral_bins)
        self.processes = processes
        self.check_config()

        if assumed_emissivity is None:
            assumed_emissivity = band_normal_emissivity(
                self.materials[self.surface_material], self.wavelength_range, self.spectral_bins
            )
        self.assumed_emissivity = float(assumed_emissivity)
        if not 0.0 < self.assumed_emissivity <= 1.0:
            raise ValueError(f"assumed_emissivity must be in (0, 1], got {self.assumed_emissivity}")
        self.result: CameraImages | None = None

    def check_config(self) -> None:
        """Raise ValueError for a config raysect cannot use, before any long run starts."""
        if not 0.0 < self.wavelength_range[0] < self.wavelength_range[1]:
            raise ValueError(
                f"wavelength_range must be (min, max) with 0 < min < max [nm], "
                f"got {self.wavelength_range}"
            )
        check_camera(self.camera)
        for name, spec in self.materials.items():
            check_material(name, spec, self.wavelength_range)
        self.check_material_name(self.surface_material, "surface_material")
        for index, component in enumerate(self.scene):
            self.check_scene_component(index, component)

        if self.max_reflections < 0:
            raise ValueError(f"max_reflections must be at least 0, got {self.max_reflections}")
        if self.pixel_samples < 1 or self.spectral_bins < 1:
            raise ValueError("pixel_samples and spectral_bins must be at least 1")
        if self.processes is not None and self.processes < 1:
            raise ValueError(f"processes must be None or at least 1, got {self.processes}")

    def check_material_name(self, name: str, where: str) -> None:
        if name not in self.materials:
            raise ValueError(
                f"{where}: unknown material {name!r}; known materials: {sorted(self.materials)}"
            )

    def check_scene_component(self, index: int, component: Mapping[str, Any]) -> None:
        where = f"OpticalModel scene component {index}"
        if not isinstance(component, Mapping):
            raise ValueError(f"{where} must be a dict, got {component!r}")
        check_config(component, required=["mesh", "material", "temperature"], where=where)
        if not Path(component["mesh"]).is_file():
            raise ValueError(f"{where}: mesh file not found: {component['mesh']}")
        self.check_material_name(component["material"], where)
        temperature = float(component["temperature"])
        if not (np.isfinite(temperature) and temperature > 0.0):
            raise ValueError(f"{where}: temperature must be finite and above 0 K")

    def run(self, step_dir: Path, temperature: TemperatureField) -> None:
        """Render one camera image per time of ``temperature``."""
        if not np.all(np.isfinite(temperature.temperature)) or temperature.temperature.min() <= 0:
            raise ValueError("Raysect: surface temperatures must be finite and above 0 K")

        world = World()
        triangles = non_degenerate_triangles(temperature.mesh)
        surface_temperature = NodeTemperature(triangles)
        material = make_material(self.materials[self.surface_material], surface_temperature)
        add_mesh(world, temperature.mesh, triangles, material)

        for component in self.scene:
            mesh, _, _ = read_triangle_mesh(component["mesh"])
            material = make_material(
                self.materials[component["material"]],
                ConstantTemperature(component["temperature"]),
            )
            add_mesh(world, mesh, non_degenerate_triangles(mesh), material)

        pipeline = RadiancePipeline2D(display_progress=False, accumulate=False)
        camera = make_camera(self.camera, pipeline, world)
        self.set_render_options(camera)

        radiance = []
        for index, time in enumerate(temperature.times):
            print(f"Raysect: rendering image {index + 1}/{len(temperature.times)} (t = {time} s)")
            surface_temperature.values = temperature.temperature[index]
            radiance.append(self.render(camera, pipeline))

        # The same spectral bins as the rendering, so that a grey body at T gives back T
        bins = Spectrum(*self.wavelength_range, self.spectral_bins)
        self.result = CameraImages(
            times=temperature.times,
            radiance=np.array(radiance),
            wavelength_range=self.wavelength_range,
            apparent_temperature=apparent_temperature(
                np.array(radiance), bins.wavelengths, bins.delta_wavelength, self.assumed_emissivity
            ),
            assumed_emissivity=self.assumed_emissivity,
        )

    def render(self, camera: Observer2D, pipeline: RadiancePipeline2D) -> np.ndarray:
        """Render one image; return its radiance [row, column], row 0 = image top."""
        # Each reflection nests a few Python calls (the materials are Python classes), so
        # Python's default recursion limit (1000) would stop paths at about 300 reflections.
        recursion_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(recursion_limit, 1000 + 4 * self.max_reflections))
        try:
            camera.observe()
        finally:
            sys.setrecursionlimit(recursion_limit)
        # raysect frames are indexed [x, y] with x = 0 left and y = 0 top
        return np.array(pipeline.frame.mean.T)

    def set_render_options(self, camera: Observer2D) -> None:
        """Rays per pixel, spectral bins, reflection depth and CPU processes of ``camera``."""
        camera.pixel_samples = self.pixel_samples
        camera.spectral_bins = self.spectral_bins
        # Set the maximum first: raysect's default band (375-740 nm) lies below the IR band.
        camera.max_wavelength = self.wavelength_range[1]
        camera.min_wavelength = self.wavelength_range[0]
        # The camera ray has depth 0 and each reflection adds 1; rays at max depth return
        # nothing. No Russian roulette (extinction), so every path goes up to max_reflections;
        # raysect needs a minimum extinction depth between 1 and the maximum depth.
        camera.ray_extinction_prob = 0.0
        camera.ray_extinction_min_depth = 1
        camera.ray_max_depth = self.max_reflections + 1
        if self.processes == 1:
            camera.render_engine = SerialEngine()
        else:
            camera.render_engine = MulticoreEngine(processes=self.processes)
        camera.quiet = True

    def get_result(self) -> CameraImages:
        """One radiance and apparent temperature image per time."""
        if self.result is None:
            raise RuntimeError("Raysect backend has not been run yet.")
        return self.result
