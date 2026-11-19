"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

import irflow.code_paths
from irflow.data import CameraImages, HeatFluxField, SurfaceMesh, TemperatureField


@pytest.fixture(autouse=True)
def no_user_code_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Make sure tests never read the user's real code-paths file."""
    monkeypatch.delenv(irflow.code_paths.CODE_PATHS_ENV_VAR, raising=False)
    monkeypatch.setattr(
        irflow.code_paths, "DEFAULT_CODE_PATHS_FILE", tmp_path / "missing_code_paths.toml"
    )


@pytest.fixture
def openfoam_bashrc(tmp_path: Path) -> Path:
    """An empty file standing in for the OpenFOAM bashrc."""
    bashrc = tmp_path / "OpenFOAM" / "etc" / "bashrc"
    bashrc.parent.mkdir(parents=True)
    bashrc.touch()
    return bashrc


@pytest.fixture
def square_mesh() -> SurfaceMesh:
    """A unit square in the z=0 plane split into two triangles."""
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    return SurfaceMesh(nodes=nodes, triangles=np.array([[0, 1, 2], [0, 2, 3]]))


@pytest.fixture
def heat_flux_field(square_mesh: SurfaceMesh) -> HeatFluxField:
    return HeatFluxField(mesh=square_mesh, heat_flux=np.array([1.0e6, 2.0e6]))


@pytest.fixture
def temperature_field(square_mesh: SurfaceMesh) -> TemperatureField:
    temperature = np.array([[300.0, 300.0, 300.0, 300.0], [310.0, 320.0, 330.0, 340.0]])
    return TemperatureField(mesh=square_mesh, times=np.array([0.0, 0.5]), temperature=temperature)


@pytest.fixture
def camera_images() -> CameraImages:
    """Two images of 3 rows x 4 columns, every pixel with a different value."""
    radiance = np.arange(24.0).reshape(2, 3, 4)
    return CameraImages(
        times=np.array([0.0, 0.5]), radiance=radiance, wavelength_range=(3000.0, 5000.0)
    )


@pytest.fixture
def gyrohit_target_vtk(tmp_path: Path, heat_flux_field: HeatFluxField) -> Path:
    """Legacy VTK unstructured grid like GYROHIT output, with cell array ``Heat-flux[W/m^2]``."""
    mesh = heat_flux_field.mesh
    cells = np.hstack([np.full((mesh.n_cells, 1), 3), mesh.triangles]).ravel()
    grid = pv.UnstructuredGrid(cells, np.full(mesh.n_cells, pv.CellType.TRIANGLE), mesh.nodes)
    grid.cell_data["Heat-flux[W/m^2]"] = heat_flux_field.heat_flux

    path = tmp_path / "target.vtk"
    grid.save(path, binary=False)
    return path
