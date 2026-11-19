"""Saving and loading step results in different file formats.

The user script chooses formats with ``IRFlow(..., save_formats=["vtk"])``. To add a format
(e.g. HDF5 or MED), add a module next to ``vtk.py`` with the same functions and register it
in :data:`SUPPORTED_FORMATS` and the dispatch below.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from irflow.data import CameraImages, HeatFluxField, TemperatureField
from irflow.io import vtk

SUPPORTED_FORMATS = ("vtk",)

HEAT_FLUX_SUFFIXES = {".vtk": vtk, ".vtp": vtk, ".vtu": vtk}
TEMPERATURE_SUFFIXES = {".pvd": vtk}
IMAGES_SUFFIXES = {".pvd": vtk}


def check_formats(formats: Sequence[str]) -> tuple[str, ...]:
    """Return the formats in lower case; raise ValueError for unsupported ones."""
    if isinstance(formats, str):
        raise TypeError(f"formats must be a list of names, e.g. ['{formats}']")
    checked = tuple(fmt.lower() for fmt in formats)
    unknown = [fmt for fmt in checked if fmt not in SUPPORTED_FORMATS]
    if unknown:
        raise ValueError(f"Unsupported save formats {unknown}; supported: {SUPPORTED_FORMATS}")
    return checked


def save(
    field: HeatFluxField | TemperatureField | CameraImages,
    directory: str | Path,
    formats: Sequence[str],
) -> list[Path]:
    """Save a step result into ``directory`` in each of ``formats``; return the written files."""
    if not isinstance(field, (HeatFluxField, TemperatureField, CameraImages)):
        raise TypeError(f"Cannot save a {type(field).__name__}")

    paths = []
    for fmt in check_formats(formats):
        if fmt == "vtk":
            if isinstance(field, HeatFluxField):
                paths.append(vtk.save_heat_flux(field, Path(directory)))
            elif isinstance(field, TemperatureField):
                paths.append(vtk.save_temperature(field, Path(directory)))
            else:
                paths.append(vtk.save_images(field, Path(directory)))
    return paths


def load_heat_flux(path: str | Path, array_name: str = "heat_flux") -> HeatFluxField:
    """Load a heat flux field; the format follows from the file suffix."""
    module = HEAT_FLUX_SUFFIXES.get(Path(path).suffix.lower())
    if module is None:
        raise ValueError(
            f"Unknown heat flux file type {path}; supported: {list(HEAT_FLUX_SUFFIXES)}"
        )
    return module.load_heat_flux(path, array_name=array_name)


def load_temperature(path: str | Path) -> TemperatureField:
    """Load a temperature field; the format follows from the file suffix."""
    module = TEMPERATURE_SUFFIXES.get(Path(path).suffix.lower())
    if module is None:
        raise ValueError(
            f"Unknown temperature file type {path}; supported: {list(TEMPERATURE_SUFFIXES)}"
        )
    return module.load_temperature(path)


def load_images(path: str | Path) -> CameraImages:
    """Load camera images; the format follows from the file suffix."""
    module = IMAGES_SUFFIXES.get(Path(path).suffix.lower())
    if module is None:
        raise ValueError(f"Unknown image file type {path}; supported: {list(IMAGES_SUFFIXES)}")
    return module.load_images(path)
