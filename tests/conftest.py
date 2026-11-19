"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

import irflow.code_paths


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
def two_triangles() -> dict[str, np.ndarray]:
    """A unit square in the z=0 plane split into two triangles, with heat flux per triangle."""
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    cells = np.array([[0, 1, 2], [0, 2, 3]])
    heat_flux = np.array([1.0e6, 2.0e6])
    return {"nodes": nodes, "cells_2d": cells, "heat_flux": heat_flux}


@pytest.fixture
def gyrohit_target_vtk(tmp_path: Path, two_triangles: dict[str, np.ndarray]) -> Path:
    """Legacy VTK unstructured grid like GYROHIT output, with cell array ``Heat-flux[W/m^2]``."""
    n_cells = len(two_triangles["cells_2d"])
    cells = np.hstack([np.full((n_cells, 1), 3), two_triangles["cells_2d"]]).ravel()
    grid = pv.UnstructuredGrid(
        cells, np.full(n_cells, pv.CellType.TRIANGLE), two_triangles["nodes"]
    )
    grid.cell_data["Heat-flux[W/m^2]"] = two_triangles["heat_flux"]

    path = tmp_path / "target.vtk"
    grid.save(path, binary=False)
    return path
