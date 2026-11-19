"""Tests for config checks: clear errors for missing keys, typos and bad time stepping."""

from __future__ import annotations

from pathlib import Path

import pytest

from irflow import FieldLineTracing, ThermalModel
from irflow.config import check_config, check_time_stepping


def test_valid_config_passes() -> None:
    check_config({"a": 1, "b": 2}, required=["a"], optional=["b", "c"])


def test_missing_and_unknown_keys_are_all_reported() -> None:
    with pytest.raises(ValueError) as error:
        check_config({"mesh_fiel": "x", "colour": 1}, required=["mesh_file", "mesh_type"])
    message = str(error.value)
    assert "missing key 'mesh_file'" in message
    assert "missing key 'mesh_type'" in message
    assert "unknown key 'mesh_fiel' (did you mean 'mesh_file'?)" in message
    assert "unknown key 'colour'" in message


@pytest.mark.parametrize(
    "time_stepping",
    [
        {"type": "steady"},
        {"type": "Steady", "time_step": None, "end_time": None},
        {"type": "transient", "time_step": 0.1, "end_time": 0.5},
    ],
)
def test_valid_time_stepping(time_stepping: dict) -> None:
    check_time_stepping(time_stepping)


@pytest.mark.parametrize(
    "time_stepping, match",
    [
        ({"type": "unsteady"}, "'steady' or 'transient'"),
        ({"type": "transient", "time_step": 0.1}, "missing key 'end_time'"),
        ({"type": "transient", "time_step": -0.1, "end_time": 1.0}, "positive"),
        ({"type": "transient", "time_step": 2.0, "end_time": 1.0}, "larger than end_time"),
    ],
)
def test_invalid_time_stepping(time_stepping: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        check_time_stepping(time_stepping)


def test_thermal_model_reports_typo_before_running(openfoam_bashrc: Path) -> None:
    config = {
        "mesh_file": "mesh.unv",
        "mesh_type": "unv",
        "plasma_bc_nmae": "plasmaBC",
        "openfoam_bashrc": str(openfoam_bashrc),
    }
    with pytest.raises(ValueError, match="did you mean 'plasma_bc_name'"):
        ThermalModel(backend_code="openfoam", config=config)


def test_thermal_model_checks_time_stepping(openfoam_bashrc: Path) -> None:
    config = {
        "mesh_file": "mesh.unv",
        "mesh_type": "unv",
        "plasma_bc_name": "plasmaBC",
        "openfoam_bashrc": str(openfoam_bashrc),
        "time_stepping": {"type": "transient", "time_step": 0.1},
    }
    with pytest.raises(ValueError, match="end_time"):
        ThermalModel(backend_code="openfoam", config=config)


def test_gyrohit_only_needs_target(gyrohit_target_vtk: Path) -> None:
    config = {"target_mesh_path": str(gyrohit_target_vtk), "target_mesh_type": "vtk"}
    FieldLineTracing(backend_code="gyrohit", config=config)
