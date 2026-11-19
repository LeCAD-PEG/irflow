"""Tests for the field line tracing step with the GYROHIT backend (reads precomputed flux)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from irflow import FieldLineTracing
from irflow.data import HeatFluxField


def gyrohit_config(target: Path, shadow: str | None = None) -> dict[str, str | None]:
    return {
        "target_mesh_path": str(target),
        "target_mesh_type": "vtk",
        "shadow_mesh_path": shadow,
        "shadow_mesh_type": None,
        "equilibrium_path": None,
        "equilibrium_type": None,
    }


def test_gyrohit_result_is_heat_flux_on_target(
    tmp_path: Path, gyrohit_target_vtk: Path, heat_flux_field: HeatFluxField
) -> None:
    flt = FieldLineTracing(backend_code="gyrohit", config=gyrohit_config(gyrohit_target_vtk))
    flt.run(tmp_path)

    result = flt.get_irflow_pass_data()

    np.testing.assert_allclose(result.heat_flux, heat_flux_field.heat_flux)
    np.testing.assert_allclose(result.mesh.nodes, heat_flux_field.mesh.nodes)
    np.testing.assert_array_equal(result.mesh.triangles, heat_flux_field.mesh.triangles)


def test_no_result_before_run(gyrohit_target_vtk: Path) -> None:
    flt = FieldLineTracing(backend_code="gyrohit", config=gyrohit_config(gyrohit_target_vtk))
    with pytest.raises(RuntimeError, match="no result yet"):
        flt.get_irflow_pass_data()


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
    assert flt.has_own_input
    with pytest.raises(ValueError):
        flt.set_irflow_pass_data(None)
