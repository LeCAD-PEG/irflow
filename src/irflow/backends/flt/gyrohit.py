"""Field line tracing backend for GYROHIT.

For now this only reads heat flux that GYROHIT has already computed on the target mesh
(VTK file with cell array ``Heat-flux[W/m^2]``); it does not run GYROHIT itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

HEAT_FLUX_ARRAY = "Heat-flux[W/m^2]"


def read_triangle_vtk(path: str | Path) -> dict[str, Any]:
    """Read a triangular surface mesh from VTK: ``{"nodes", "cells_2d", "cell_data"}``."""
    mesh = pv.read(path)
    if isinstance(mesh, pv.PolyData):
        cells = mesh.regular_faces
    else:
        if set(mesh.celltypes) != {pv.CellType.TRIANGLE}:
            raise ValueError(f"Expected only triangles in {path}")
        cells = mesh.cells_dict[pv.CellType.TRIANGLE]

    return {
        "nodes": np.asarray(mesh.points, dtype=float),
        "cells_2d": np.asarray(cells, dtype=int),
        "cell_data": {name: np.asarray(mesh.cell_data[name]) for name in mesh.cell_data},
    }


class BackendGyrohit:
    """Connector between the irflow field line tracing step and GYROHIT results."""

    def __init__(
        self,
        target_mesh_path: str,
        shadow_mesh_path: str | None,
        equilibrium_path: str | None,
        target_mesh_type: str,
        shadow_mesh_type: str | None,
        equilibrium_type: str | None,
    ) -> None:
        self.target_mesh_path = target_mesh_path
        self.target_mesh_type = target_mesh_type
        self.shadow_mesh_path = shadow_mesh_path
        self.shadow_mesh_type = shadow_mesh_type
        self.equilibrium_path = equilibrium_path
        self.equilibrium_type = equilibrium_type

        # irflow data format (to be passed back to FLT class)
        self.target_irflow: dict[str, Any] | None = None

    def get_target_mesh_irflow(self) -> dict[str, Any]:
        """Return the target mesh ``{"nodes": (N, 3), "cells_2d": (M, 3)}``."""
        if self.target_irflow is None:
            if self.target_mesh_type.lower() != "vtk":
                raise ValueError(
                    f"Unknown target mesh type '{self.target_mesh_type}'. Supported: 'vtk'."
                )
            data = read_triangle_vtk(self.target_mesh_path)
            self.target_irflow = {
                "nodes": data["nodes"],
                "cells_2d": data["cells_2d"],
                "heat_flux": data["cell_data"].get(HEAT_FLUX_ARRAY),
            }
        return self.target_irflow

    def run(self, dir_path: Path) -> None:
        """Read the precomputed GYROHIT heat flux from the target mesh file."""
        if self.shadow_mesh_path is not None or self.equilibrium_path is not None:
            raise NotImplementedError(
                "Running GYROHIT is not supported yet; give only a target mesh with "
                f"precomputed '{HEAT_FLUX_ARRAY}' (shadow and equilibrium must be None)."
            )

        print("Gyrohit: processing heat fluxes in target mesh.")
        if self.get_target_mesh_irflow()["heat_flux"] is None:
            raise KeyError(f"No cell array '{HEAT_FLUX_ARRAY}' in {self.target_mesh_path}")

    def get_irflow_data(self) -> dict[str, Any]:
        target = self.get_target_mesh_irflow()
        return {"target": {"nodes": target["nodes"], "cells_2d": target["cells_2d"]}}

    def get_heat_flux(self) -> np.ndarray:
        """Heat flux on each target triangle [W/m^2]."""
        return self.get_target_mesh_irflow()["heat_flux"]
