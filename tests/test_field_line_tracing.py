"""Tests for the field line tracing step with the GYROHIT backend (reads precomputed flux)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from irflow import FieldLineTracing
from irflow.backends.flt.gyrohit import read_triangle_vtk


def gyrohit_config(target: Path, shadow: str | None = None) -> dict[str, str | None]:
    return {
        "target_mesh_path": str(target),
        "target_mesh_type": "vtk",
        "shadow_mesh_path": shadow,
        "shadow_mesh_type": None,
        "equilibrium_path": None,
        "equilibrium_type": None,
    }


def test_read_triangle_vtk(gyrohit_target_vtk: Path, two_triangles: dict) -> None:
    data = read_triangle_vtk(gyrohit_target_vtk)
    np.testing.assert_allclose(data["nodes"], two_triangles["nodes"])
    np.testing.assert_array_equal(data["cells_2d"], two_triangles["cells_2d"])
    np.testing.assert_allclose(data["cell_data"]["Heat-flux[W/m^2]"], two_triangles["heat_flux"])


def test_gyrohit_pass_data_is_flux_at_cell_centres(
    tmp_path: Path, gyrohit_target_vtk: Path, two_triangles: dict
) -> None:
    flt = FieldLineTracing(backend_code="gyrohit", config=gyrohit_config(gyrohit_target_vtk))
    flt.run(tmp_path)

    data = flt.get_irflow_pass_data()

    expected_centres = np.array([[2 / 3, 1 / 3, 0.0], [1 / 3, 2 / 3, 0.0]])
    np.testing.assert_allclose(data["xyz"], expected_centres)
    np.testing.assert_allclose(data["heat_flux"], two_triangles["heat_flux"])
    np.testing.assert_allclose(flt.result["heat_flux"], two_triangles["heat_flux"])


def test_gyrohit_cannot_run_the_code_itself(tmp_path: Path, gyrohit_target_vtk: Path) -> None:
    config = gyrohit_config(gyrohit_target_vtk, shadow="shadow.vtk")
    flt = FieldLineTracing(backend_code="gyrohit", config=config)
    with pytest.raises(NotImplementedError):
        flt.run(tmp_path)


def test_run_needs_existing_directory(tmp_path: Path, gyrohit_target_vtk: Path) -> None:
    flt = FieldLineTracing(backend_code="gyrohit", config=gyrohit_config(gyrohit_target_vtk))
    with pytest.raises(FileNotFoundError):
        flt.run(tmp_path / "missing")


def test_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unknown field line tracing backend"):
        FieldLineTracing(backend_code="nope")


def test_first_step_takes_no_upstream_data(gyrohit_target_vtk: Path) -> None:
    flt = FieldLineTracing(backend_code="gyrohit", config=gyrohit_config(gyrohit_target_vtk))
    with pytest.raises(ValueError):
        flt.set_irflow_pass_data({})
