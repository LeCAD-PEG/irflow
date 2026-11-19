"""Tests for the surface overlap check used before mapping heat flux."""

from __future__ import annotations

import numpy as np
import pytest

from irflow.data import SurfaceMesh
from irflow.mapping import check_overlap, distances_to_surface


def grid_mesh(n: int = 10, size: float = 1.0) -> SurfaceMesh:
    """A flat square of side ``size`` in the z=0 plane, split into 2 * n * n triangles."""
    x, y = np.meshgrid(np.linspace(0.0, size, n + 1), np.linspace(0.0, size, n + 1))
    nodes = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    triangles = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            triangles += [[a, a + 1, a + n + 2], [a, a + n + 2, a + n + 1]]
    return SurfaceMesh(nodes=nodes, triangles=np.array(triangles))


def transformed(
    mesh: SurfaceMesh, offset: tuple[float, float, float] = (0.0, 0.0, 0.0), scale: float = 1.0
) -> SurfaceMesh:
    return SurfaceMesh(nodes=mesh.nodes * scale + np.array(offset), triangles=mesh.triangles)


def test_distance_to_surface(square_mesh: SurfaceMesh) -> None:
    points = np.array([[0.5, 0.5, 2.0], [0.2, 0.7, 0.0], [3.0, 0.5, 0.0]])
    np.testing.assert_allclose(distances_to_surface(square_mesh, points), [2.0, 0.0, 2.0])


def test_same_geometry_different_resolution_passes() -> None:
    assert check_overlap(grid_mesh(n=20), grid_mesh(n=3)) == pytest.approx(0.0, abs=1e-12)
    assert check_overlap(grid_mesh(n=3), grid_mesh(n=20)) == pytest.approx(0.0, abs=1e-12)


def test_small_offset_within_default_tolerance() -> None:
    mesh = grid_mesh()
    assert check_overlap(mesh, transformed(mesh, (0.0, 0.0, 0.01))) == pytest.approx(0.01)


def test_shifted_surface_fails() -> None:
    mesh = grid_mesh()
    with pytest.raises(ValueError, match="away from the heat flux surface"):
        check_overlap(mesh, transformed(mesh, (0.0, 0.0, 0.5)))


def test_millimetres_instead_of_metres_fails() -> None:
    mesh = grid_mesh()
    with pytest.raises(ValueError, match="metres"):
        check_overlap(mesh, transformed(mesh, scale=1000.0))


def test_explicit_tolerance() -> None:
    mesh = grid_mesh()
    target = transformed(mesh, (0.0, 0.0, 0.01))
    with pytest.raises(ValueError):
        check_overlap(mesh, target, tolerance=0.001)
    assert check_overlap(mesh, target, tolerance=np.inf) == pytest.approx(0.01)
