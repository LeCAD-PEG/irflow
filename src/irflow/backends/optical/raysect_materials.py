"""Surface materials for the raysect optical backend: thermal emission plus reflection.

Every material is opaque and two-sided (both faces of a triangle emit and reflect). It emits
at the local surface temperature, which comes from a temperature source shared by all material
types: :class:`NodeTemperature` (thermal model result, interpolated inside each triangle) or
:class:`ConstantTemperature` (machine components of the ``scene``).

Material specs (config dicts, by ``type``):

- ``{"type": "tungsten"}``: smooth metal. Mirror reflection with the Fresnel reflectance R of
  raysect's tungsten optical constants n, k (Rakic et al. 1998, room temperature, 0.2-12.4 um);
  emissivity 1 - R for the same wavelength and angle (Kirchhoff). A rough-surface option
  (microfacets) could be added later.
- ``{"type": "grey", "emissivity": e, "reflection": "diffuse" | "specular" | "none"}``: grey
  body. Reflectivity 1 - e, Lambertian (``diffuse``) or mirror-like (``specular``); ``none``
  ignores reflection (emission only, the reflectivity is lost).

Kirchhoff's law (emissivity = 1 - reflectivity for an opaque body) makes emission and
reflection consistent: inside a closed isothermal enclosure every material looks like a black
body. Constructions:

- specular (tungsten, grey specular): :class:`SpecularSurface`, one Python material that
  traces the mirror ray and adds R x reflected + (1 - R) x Planck, with the same R array.
- grey diffuse: raysect's ``Lambert(1 - e)`` plus a :class:`GreyEmitter` of emissivity e,
  summed with raysect's ``Add`` material.
- grey none: a :class:`GreyEmitter` alone.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Union

import numpy as np
import raysect.optical.library.metal
from raysect.core import AffineMatrix3D, Intersection, Normal3D, Point3D, Primitive
from raysect.optical import ConstantSF, InterpolatedSF, Ray, Spectrum, World
from raysect.optical.material import Add, Lambert, Material, NullVolume

from irflow.backends.optical.planck import planck_radiance
from irflow.config import check_config

# Built-in materials; the user's ``materials`` config is merged over these.
MATERIAL_LIBRARY: dict[str, dict[str, Any]] = {"tungsten": {"type": "tungsten"}}

# Config keys per material type: (required, optional), besides "type"
MATERIAL_KEYS: dict[str, tuple[list[str], list[str]]] = {
    "tungsten": ([], []),
    "grey": (["emissivity", "reflection"], []),
}
REFLECTION_TYPES = ("diffuse", "specular", "none")

TUNGSTEN_DATA = Path(raysect.optical.library.metal.__file__).parent / "data" / "tungsten.json"


class NodeTemperature:
    """Temperature at the nodes of a raysect ``Mesh``, interpolated linearly inside triangles.

    ``triangles`` (M, 3) must be the triangles of the raysect ``Mesh`` in the same order, so
    that the triangle index raysect reports for a hit points into it. Set ``values`` (N,) [K]
    before rendering.
    """

    def __init__(self, triangles: np.ndarray) -> None:
        self.triangles = triangles
        self.values = np.zeros(0)

    def at(self, intersection: Intersection) -> float:
        """Temperature [K] at a hit point: barycentric interpolation of the node values.

        Raysect's ``MeshIntersection`` holds the triangle index and the barycentric coordinates
        (u, v, w) of the hit point, which weight the triangle's three nodes in order.
        """
        nodes = self.triangles[intersection.triangle]
        weights = (intersection.u, intersection.v, intersection.w)
        return float(np.dot(weights, self.values[nodes]))


class ConstantTemperature:
    """The same temperature [K] everywhere on a surface."""

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def at(self, intersection: Intersection) -> float:
        return self.value


TemperatureSource = Union[NodeTemperature, ConstantTemperature]


class GreyEmitter(NullVolume):
    """Grey thermal emission: emissivity x Planck spectral radiance at the local temperature."""

    def __init__(self, temperature: TemperatureSource, emissivity: float) -> None:
        super().__init__()
        self.temperature = temperature
        self.emissivity = emissivity

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
        """Spectral radiance [W/(m^2 sr nm)] emitted at the hit point."""
        spectrum = ray.new_spectrum()
        temperature = self.temperature.at(intersection)
        spectrum.samples[:] = self.emissivity * planck_radiance(spectrum.wavelengths, temperature)
        return spectrum


class GreyReflectance:
    """The same reflectance at every wavelength and angle."""

    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, ray: Ray, cos_incidence: float) -> np.ndarray:
        return np.full(ray.bins, self.value)


class FresnelReflectance:
    """Fresnel reflectance of a smooth surface of complex refractive index n + ik, from vacuum.

    Unpolarised light: the mean of the s- and p-polarised reflectances, exact Fresnel equations.
    ``index`` and ``extinction`` are raysect spectral functions of wavelength [nm]; they are
    averaged over each spectral bin of the ray, as raysect does for its own materials.
    """

    def __init__(self, index: InterpolatedSF, extinction: InterpolatedSF) -> None:
        self.index = index
        self.extinction = extinction
        self.refractive_index: dict[tuple[float, float, int], np.ndarray] = {}

    def __call__(self, ray: Ray, cos_incidence: float) -> np.ndarray:
        band = (ray.min_wavelength, ray.max_wavelength, ray.bins)
        if band not in self.refractive_index:
            n = self.index.sample(*band)
            k = self.extinction.sample(*band)
            self.refractive_index[band] = n + 1j * k
        return fresnel_reflectance(self.refractive_index[band], cos_incidence)


def fresnel_reflectance(refractive_index: np.ndarray, cos_incidence: float) -> np.ndarray:
    """Unpolarised reflectance from vacuum onto a medium of complex ``refractive_index``."""
    permittivity = refractive_index**2
    # n cos(theta_t), on the branch with positive imaginary part (wave decays into the medium)
    root = np.sqrt(permittivity - (1.0 - cos_incidence**2))
    r_s = (cos_incidence - root) / (cos_incidence + root)
    r_p = (permittivity * cos_incidence - root) / (permittivity * cos_incidence + root)
    return np.asarray(0.5 * (np.abs(r_s) ** 2 + np.abs(r_p) ** 2))


Reflectance = Union[GreyReflectance, FresnelReflectance]


class SpecularSurface(NullVolume):
    """Opaque mirror-like surface with Kirchhoff emission: R x reflected + (1 - R) x Planck.

    ``reflectance(ray, cos_incidence)`` gives R for each spectral bin of the ray. The reflected
    ray is traced like raysect's ``Conductor`` does it, in the primitive's local frame.
    """

    def __init__(self, temperature: TemperatureSource, reflectance: Reflectance) -> None:
        super().__init__()
        self.temperature = temperature
        self.reflectance = reflectance

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
        """Spectral radiance [W/(m^2 sr nm)] leaving the hit point along the reverse ray."""
        incident = ray.direction.transform(world_to_primitive).normalise()
        normal = normal.normalise()
        cos_signed = normal.dot(incident)
        reflected = (incident - normal * (2.0 * cos_signed)).transform(primitive_to_world)

        # The reflected ray starts on the side of the surface the incident ray came from.
        start = inside_point if cos_signed > 0.0 else outside_point
        spectrum = ray.spawn_daughter(start.transform(primitive_to_world), reflected).trace(world)

        reflectance = self.reflectance(ray, abs(cos_signed))
        emission = planck_radiance(spectrum.wavelengths, self.temperature.at(intersection))
        spectrum.samples[:] = reflectance * spectrum.samples + (1.0 - reflectance) * emission
        return spectrum


@lru_cache(maxsize=1)
def tungsten_optical_constants() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raysect's tungsten data: wavelength [nm], refractive index n and extinction k."""
    data = json.loads(TUNGSTEN_DATA.read_text())
    return (
        np.array(data["wavelength"], dtype=float),
        np.array(data["index"], dtype=float),
        np.array(data["extinction"], dtype=float),
    )


def check_material(
    name: str, spec: Mapping[str, Any], wavelength_range: tuple[float, float]
) -> None:
    """Raise ValueError if material ``spec`` is invalid for the camera band [nm]."""
    where = f"OpticalModel material '{name}'"
    if not isinstance(spec, Mapping):
        raise ValueError(f"{where} must be a dict with a 'type', got {spec!r}")
    kind = spec.get("type")
    if kind not in MATERIAL_KEYS:
        raise ValueError(f"{where}: 'type' must be one of {sorted(MATERIAL_KEYS)}, got {kind!r}")
    required, optional = MATERIAL_KEYS[kind]
    check_config(spec, ["type", *required], optional, where=where)

    if kind == "tungsten":
        wavelength = tungsten_optical_constants()[0]
        if wavelength_range[0] < wavelength[0] or wavelength_range[1] > wavelength[-1]:
            raise ValueError(
                f"{where}: tungsten optical data cover {wavelength[0]}-{wavelength[-1]} nm, "
                f"the camera band {wavelength_range} nm lies outside"
            )
    else:
        emissivity = float(spec["emissivity"])
        if not 0.0 < emissivity <= 1.0:
            raise ValueError(f"{where}: emissivity must be in (0, 1], got {emissivity}")
        if spec["reflection"] not in REFLECTION_TYPES:
            raise ValueError(
                f"{where}: reflection must be one of {REFLECTION_TYPES}, "
                f"got {spec['reflection']!r}"
            )


def band_normal_emissivity(
    spec: Mapping[str, Any], wavelength_range: tuple[float, float], bins: int
) -> float:
    """Emissivity of a material at normal incidence, averaged over the bins of the camera band.

    For a grey material this is its ``emissivity``. For tungsten it is the mean of 1 - R over
    the ``bins`` wavelength bins of ``wavelength_range`` [nm], with R the normal-incidence
    Fresnel reflectance of the same bin-averaged n, k that the material uses when rendering.
    """
    if spec["type"] != "tungsten":
        return float(spec["emissivity"])
    reflectance = tungsten_reflectance()
    refractive_index = reflectance.index.sample(*wavelength_range, bins) + 1j * (
        reflectance.extinction.sample(*wavelength_range, bins)
    )
    return float(np.mean(1.0 - fresnel_reflectance(refractive_index, 1.0)))


def tungsten_reflectance() -> FresnelReflectance:
    """Fresnel reflectance with raysect's tungsten optical constants."""
    wavelength, index, extinction = tungsten_optical_constants()
    return FresnelReflectance(
        InterpolatedSF(wavelength, index), InterpolatedSF(wavelength, extinction)
    )


def make_material(spec: Mapping[str, Any], temperature: TemperatureSource) -> Material:
    """Raysect material for a checked material ``spec`` at the given surface temperature."""
    if spec["type"] == "tungsten":
        return SpecularSurface(temperature, tungsten_reflectance())

    emissivity = float(spec["emissivity"])
    if spec["reflection"] == "none" or emissivity == 1.0:
        return GreyEmitter(temperature, emissivity)
    if spec["reflection"] == "specular":
        return SpecularSurface(temperature, GreyReflectance(1.0 - emissivity))
    return Add(
        Lambert(ConstantSF(1.0 - emissivity)),
        GreyEmitter(temperature, emissivity),
        surface_only=True,
    )
