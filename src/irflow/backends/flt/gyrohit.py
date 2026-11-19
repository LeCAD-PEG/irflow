"""Field line tracing backend for GYROHIT.

For now this only reads heat flux that GYROHIT has already computed on the target mesh
(VTK file with cell array ``Heat-flux[W/m^2]``); it does not run GYROHIT itself.
"""

from __future__ import annotations

from pathlib import Path

from irflow.backends.base import FieldLineTracingBackend
from irflow.data import HeatFluxField
from irflow.io.vtk import load_heat_flux

HEAT_FLUX_ARRAY = "Heat-flux[W/m^2]"


class BackendGyrohit(FieldLineTracingBackend):
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
        self.result: HeatFluxField | None = None

    def run(self, step_dir: Path) -> None:
        """Read the precomputed GYROHIT heat flux from the target mesh file."""
        if self.shadow_mesh_path is not None or self.equilibrium_path is not None:
            raise NotImplementedError(
                "Running GYROHIT is not supported yet; give only a target mesh with "
                f"precomputed '{HEAT_FLUX_ARRAY}' (shadow and equilibrium must be None)."
            )
        if self.target_mesh_type.lower() != "vtk":
            raise ValueError(
                f"Unknown target mesh type '{self.target_mesh_type}'. Supported: 'vtk'."
            )

        print("Gyrohit: reading heat fluxes from target mesh.")
        self.result = load_heat_flux(self.target_mesh_path, array_name=HEAT_FLUX_ARRAY)

    def get_result(self) -> HeatFluxField:
        """Heat flux on each target triangle [W/m^2]."""
        if self.result is None:
            raise RuntimeError("Gyrohit backend has not been run yet.")
        return self.result
