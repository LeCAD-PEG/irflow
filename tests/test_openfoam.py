"""Tests for the OpenFOAM thermal model backend that do not need OpenFOAM installed."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from irflow import ThermalModel
from irflow.backends.tm.openfoam.backend import (
    list_time_dirs,
    read_patch_temperature,
    resolve_openfoam_bashrc,
)
from irflow.backends.tm.openfoam.case_builder import OpenFOAMCaseBuilder
from irflow.backends.tm.openfoam.templates import (
    write_openfoam_plasma_bc,
    write_openfoam_sh_script,
)
from irflow.data import SurfaceMesh
from irflow.io.vtk import to_polydata

BOUNDARY_FILE = """FoamFile
{
    format      ascii;
    class       polyBoundaryMesh;
    object      boundary;
}
// comment

3
(
    plasmaBC
    {
        type            wall;
    }
    const_T
    {
        type            wall;
    }
    sample_to_housing
    {
        type            mappedWall;
    }
)
"""


def make_builder(tmp_path: Path) -> OpenFOAMCaseBuilder:
    return OpenFOAMCaseBuilder(
        case_root=tmp_path,
        case_name="openfoam_case",
        time_settings={"type": "steady"},
        mesh_type="unv",
        plasma_bc_name="plasmaBC",
    )


def test_region_boundary_map(tmp_path: Path) -> None:
    builder = make_builder(tmp_path)
    builder.create_directory_structure()
    poly_mesh = builder.constant_path / "sample" / "polyMesh"
    poly_mesh.mkdir(parents=True)
    (poly_mesh / "boundary").write_text(BOUNDARY_FILE)
    (builder.constant_path / "regionProperties").touch()

    assert builder.get_region_boundary_map() == {
        "sample": ["plasmaBC", "const_T", "sample_to_housing"]
    }


def test_heat_flux_file(tmp_path: Path) -> None:
    builder = make_builder(tmp_path)
    builder.create_directory_structure()
    builder.write_heat_flux(np.array([[0.0, 1.0, 2.0]]), np.array([5.0e6]))

    lines = (builder.constant_path / "data.txt").read_text().splitlines()
    assert lines[0] == "time: 0.0"
    np.testing.assert_allclose(np.array(lines[1].split(), dtype=float), [0.0, 1.0, 2.0, 5.0e6])


def test_plasma_bc_template_sets_initial_temperature() -> None:
    buffer = io.StringIO()
    write_openfoam_plasma_bc(buffer, 373.0)
    text = buffer.getvalue()
    assert "refValue        uniform 373.0;" in text
    assert "@INITIAL_TEMPERATURE@" not in text


def test_sh_script_sources_bashrc(tmp_path: Path, openfoam_bashrc: Path) -> None:
    buffer = io.StringIO()
    write_openfoam_sh_script(buffer, tmp_path / "case dir", openfoam_bashrc)
    text = buffer.getvalue()
    assert f"source {openfoam_bashrc}" in text
    assert f"cd '{tmp_path / 'case dir'}'" in text


def test_bashrc_must_be_configured() -> None:
    with pytest.raises(ValueError, match="OpenFOAM bashrc is not set"):
        resolve_openfoam_bashrc(None)


def test_bashrc_must_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_openfoam_bashrc(str(tmp_path / "missing_bashrc"))


def test_time_dirs_are_sorted_numerically(tmp_path: Path) -> None:
    for name in ["0.1", "0", "0.02", "1e-05", "constant", "system", "VTK"]:
        (tmp_path / name).mkdir()
    (tmp_path / "0.5").touch()  # a file, not a time directory

    assert [p.name for p in list_time_dirs(tmp_path)] == ["0", "1e-05", "0.02", "0.1"]


def test_read_patch_temperature_time_series(tmp_path: Path, square_mesh: SurfaceMesh) -> None:
    case = tmp_path / "openfoam_case"
    patch_dir = case / "VTK" / "sample" / "plasmaBC"
    patch_dir.mkdir(parents=True)

    for index, time in enumerate(["0", "0.1", "0.2"]):
        (case / time).mkdir()
        patch = to_polydata(square_mesh)
        patch.point_data["T"] = 300.0 + 10.0 * index + np.arange(4)
        patch.save(patch_dir / f"plasmaBC_{index}.vtk", binary=False)

    result = read_patch_temperature(case, "sample", "plasmaBC")

    np.testing.assert_allclose(result.times, [0.0, 0.1, 0.2])
    assert result.temperature.shape == (3, 4)
    np.testing.assert_allclose(result.temperature[2], [320.0, 321.0, 322.0, 323.0])
    assert (result.mesh.n_nodes, result.mesh.n_cells) == (4, 2)


def test_openfoam_backend_needs_bashrc_and_reports_no_result(openfoam_bashrc: Path) -> None:
    config = {
        "mesh_file": "mesh.unv",
        "mesh_type": "unv",
        "plasma_bc_name": "plasmaBC",
        "time_stepping": {"type": "steady"},
        "openfoam_bashrc": str(openfoam_bashrc),
    }
    tm = ThermalModel(backend_code="openfoam", config=config)
    with pytest.raises(RuntimeError, match="not been run"):
        tm.backend.get_result()
