"""Data passed between simulation steps.

All quantities are in SI units: lengths in m, time in s, temperature in K, heat flux in W/m^2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SurfaceMesh:
    """Triangular surface mesh in 3D."""

    nodes: np.ndarray  # (N, 3) node coordinates [m]
    triangles: np.ndarray  # (M, 3) node indices, 0-based

    def __post_init__(self) -> None:
        self.nodes = np.asarray(self.nodes, dtype=float)
        self.triangles = np.asarray(self.triangles, dtype=int)

        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError(f"nodes must have shape (N, 3), got {self.nodes.shape}")
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError(f"triangles must have shape (M, 3), got {self.triangles.shape}")
        if self.triangles.size and (
            self.triangles.min() < 0 or self.triangles.max() >= self.n_nodes
        ):
            raise ValueError("triangles refer to nodes that do not exist (indices must be 0-based)")

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_cells(self) -> int:
        return len(self.triangles)

    def cell_centers(self) -> np.ndarray:
        """Centroid of each triangle, shape (M, 3)."""
        return self.nodes[self.triangles].mean(axis=1)

    def max_edge_length(self) -> float:
        """Length of the longest triangle edge [m]."""
        corners = self.nodes[self.triangles]  # (M, 3 corners, 3 coords)
        edges = corners - np.roll(corners, 1, axis=1)
        return float(np.linalg.norm(edges, axis=2).max())


@dataclass
class HeatFluxField:
    """Heat flux on each triangle of the target mesh [W/m^2]: output of field line tracing."""

    mesh: SurfaceMesh
    heat_flux: np.ndarray  # (M,)

    def __post_init__(self) -> None:
        self.heat_flux = np.asarray(self.heat_flux, dtype=float)
        if self.heat_flux.shape != (self.mesh.n_cells,):
            raise ValueError(
                f"heat_flux must have shape ({self.mesh.n_cells},) to match the mesh cells, "
                f"got {self.heat_flux.shape}"
            )


@dataclass
class TemperatureField:
    """Temperature at the nodes of the plasma-facing surface [K]: output of the thermal model.

    There is one row per time. A steady-state result has a single time.
    """

    mesh: SurfaceMesh
    times: np.ndarray  # (Nt,) [s]
    temperature: np.ndarray  # (Nt, N) [K]

    def __post_init__(self) -> None:
        self.times = np.atleast_1d(np.asarray(self.times, dtype=float))
        self.temperature = np.asarray(self.temperature, dtype=float)

        if self.times.ndim != 1:
            raise ValueError(f"times must be one-dimensional, got shape {self.times.shape}")
        expected = (len(self.times), self.mesh.n_nodes)
        if self.temperature.shape != expected:
            raise ValueError(
                f"temperature must have shape {expected} (times, nodes), "
                f"got {self.temperature.shape}"
            )
