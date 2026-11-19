"""Tests for the thermal model step: where its heat flux input comes from."""

from __future__ import annotations

from pathlib import Path

import pytest

import irflow.io
from irflow import ThermalModel
from irflow.data import HeatFluxField, TemperatureField


@pytest.fixture
def openfoam_config(openfoam_bashrc: Path) -> dict:
    return {
        "mesh_file": "mesh.unv",
        "mesh_type": "unv",
        "plasma_bc_name": "plasmaBC",
        "time_stepping": {"type": "steady"},
        "openfoam_bashrc": str(openfoam_bashrc),
    }


def test_needs_heat_flux_before_run(tmp_path: Path, openfoam_config: dict) -> None:
    tm = ThermalModel(backend_code="openfoam", config=openfoam_config)
    assert not tm.has_own_input
    with pytest.raises(RuntimeError, match="no heat flux"):
        tm.run(tmp_path)
    with pytest.raises(RuntimeError, match="no result yet"):
        tm.get_irflow_pass_data()


def test_input_data_object(openfoam_config: dict, heat_flux_field: HeatFluxField) -> None:
    tm = ThermalModel(backend_code="openfoam", config=openfoam_config, input_data=heat_flux_field)
    assert tm.has_own_input
    assert tm.heat_flux is heat_flux_field


def test_input_data_file(
    tmp_path: Path, openfoam_config: dict, heat_flux_field: HeatFluxField
) -> None:
    [path] = irflow.io.save(heat_flux_field, tmp_path, ["vtk"])
    tm = ThermalModel(backend_code="openfoam", config=openfoam_config, input_data=path)
    assert tm.has_own_input
    assert tm.heat_flux is not None
    assert list(tm.heat_flux.heat_flux) == list(heat_flux_field.heat_flux)


def test_rejects_temperature_as_input(
    openfoam_config: dict, temperature_field: TemperatureField
) -> None:
    tm = ThermalModel(backend_code="openfoam", config=openfoam_config)
    with pytest.raises(TypeError):
        tm.set_irflow_pass_data(temperature_field)


def test_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unknown thermal model backend"):
        ThermalModel(backend_code="nope")
