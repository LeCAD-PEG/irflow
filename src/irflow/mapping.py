"""Checks for mapping data from one surface mesh to another."""

from __future__ import annotations

from typing import cast

import numpy as np

from irflow.data import SurfaceMesh
from irflow.io.vtk import to_polydata


def distances_to_surface(surface: SurfaceMesh, points: np.ndarray) -> np.ndarray:
    """Distance from each point (P, 3) to the closest point on ``surface`` [m]."""
    found = to_polydata(surface).find_closest_cell(points, return_closest_point=True)
    closest = np.atleast_2d(cast(tuple[np.ndarray, np.ndarray], found)[1])
    return np.linalg.norm(points - closest, axis=1)


def check_overlap(
    source: SurfaceMesh, target: SurfaceMesh, tolerance: float | None = None
) -> float:
    """Check that the target surface lies on the source surface.

    Used before mapping heat flux from the field line tracing target (source) onto the
    thermal model plasma surface (target). The distance from every target cell centre to the
    source surface is measured; for meshes of the same geometry it is close to zero whatever
    their resolution. A large distance usually means the meshes use different units
    (m vs mm), are shifted/rotated, or the target surface extends beyond the source.

    ``tolerance`` [m] defaults to half the longest edge of either mesh. Returns the largest
    distance; raises ValueError if it exceeds the tolerance.
    """
    if tolerance is None:
        tolerance = 0.5 * max(source.max_edge_length(), target.max_edge_length())

    max_distance = float(distances_to_surface(source, target.cell_centers()).max())
    if max_distance > tolerance:
        raise ValueError(
            f"Thermal model plasma surface is up to {max_distance:.3g} m away from the heat "
            f"flux surface (tolerance {tolerance:.3g} m). Check that both meshes use metres and "
            "the same position, or set 'mapping_tolerance' in the thermal model config."
        )
    return max_distance
