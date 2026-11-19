"""Tests for the data classes passed between steps."""

from __future__ import annotations

import numpy as np
import pytest

from irflow.data import HeatFluxField, SurfaceMesh, TemperatureField


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
    field = TemperatureField(mesh=square_mesh, times=0.0, temperature=np.zeros((1, 4)))
    assert field.times.shape == (1,)
