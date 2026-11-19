"""Tests for the data classes passed between steps."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from irflow.data import CameraImages, HeatFluxField, SurfaceMesh, TemperatureField


def test_cell_centers_and_edges(square_mesh: SurfaceMesh) -> None:
    np.testing.assert_allclose(
        square_mesh.cell_centers(), [[2 / 3, 1 / 3, 0.0], [1 / 3, 2 / 3, 0.0]]
    )
    assert square_mesh.max_edge_length() == pytest.approx(np.sqrt(2.0))
    assert (square_mesh.n_nodes, square_mesh.n_cells) == (4, 2)


def test_mesh_rejects_wrong_shapes() -> None:
    with pytest.raises(ValueError, match="nodes"):
        SurfaceMesh(nodes=np.zeros((4, 2)), triangles=np.array([[0, 1, 2]]))
    with pytest.raises(ValueError, match="triangles"):
        SurfaceMesh(nodes=np.zeros((4, 3)), triangles=np.array([[0, 1, 2, 3]]))


def test_mesh_rejects_one_based_indices() -> None:
    with pytest.raises(ValueError, match="0-based"):
        SurfaceMesh(nodes=np.zeros((3, 3)), triangles=np.array([[1, 2, 3]]))


def test_heat_flux_must_match_cells(square_mesh: SurfaceMesh) -> None:
    with pytest.raises(ValueError, match="heat_flux"):
        HeatFluxField(mesh=square_mesh, heat_flux=np.zeros(4))


def test_temperature_must_match_times_and_nodes(square_mesh: SurfaceMesh) -> None:
    with pytest.raises(ValueError, match="temperature"):
        TemperatureField(mesh=square_mesh, times=np.array([0.0, 1.0]), temperature=np.zeros((1, 4)))


def test_steady_temperature_has_one_time(square_mesh: SurfaceMesh) -> None:
    field = TemperatureField(mesh=square_mesh, times=0.0, temperature=np.zeros((1, 4)))  # type: ignore[arg-type]
    assert field.times.shape == (1,)


def images(**changes: Any) -> CameraImages:
    """One image of 3 rows x 4 columns, with ``changes`` to the constructor arguments."""
    arguments: dict[str, Any] = {
        "times": [0.0],
        "radiance": np.zeros((1, 3, 4)),
        "wavelength_range": (3e3, 5e3),
        "apparent_temperature": np.full((1, 3, 4), np.nan),
        "assumed_emissivity": 1.0,
    }
    return CameraImages(**{**arguments, **changes})


def test_images_must_match_times() -> None:
    with pytest.raises(ValueError, match="radiance"):
        images(times=[0.0, 1.0])
    with pytest.raises(ValueError, match="radiance"):
        images(radiance=np.zeros((3, 4)))
    with pytest.raises(ValueError, match="apparent_temperature"):
        images(apparent_temperature=np.zeros((1, 4, 3)))


def test_images_need_increasing_wavelength_range() -> None:
    with pytest.raises(ValueError, match="wavelength_range"):
        images(wavelength_range=(5e3, 3e3))
    assert images(times=0.0, wavelength_range=[3e3, 5e3]).wavelength_range == (3000.0, 5000.0)


@pytest.mark.parametrize("emissivity", [0.0, 1.5])
def test_images_need_valid_assumed_emissivity(emissivity: float) -> None:
    with pytest.raises(ValueError, match="assumed_emissivity"):
        images(assumed_emissivity=emissivity)
