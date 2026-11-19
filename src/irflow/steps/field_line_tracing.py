"""Simulation step 1: field line tracing, giving the heat flux on the target mesh."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from irflow.backends.flt.gyrohit import BackendGyrohit
    from irflow.backends.flt.riptrace import BackendRiptrace


class FieldLineTracing:
    """Field line tracing step. The backend code is chosen with ``backend_code``."""

    dir_name = "FieldLineTracing"

    def __init__(
        self,
        name: str = "flt_case",
        enabled: bool = True,
        backend_code: str = "riptrace",
        config: dict[str, Any] | None = None,
        resources: Any = None,
    ) -> None:
        self.name = name
        self.enabled = enabled
        self.backend_code = backend_code
        self.config = config or {}
        self.resources = resources
        self.result: dict[str, Any] | None = None

        self.backend = self.get_backend(backend_code)

    def get_backend(self, backend_code: str) -> BackendRiptrace | BackendGyrohit:
        """Create the backend for ``backend_code`` from the step config."""
        # Backends are imported here so that only the chosen code has to be installed.
        if backend_code.lower() == "riptrace":
            from irflow.backends.flt.riptrace import BackendRiptrace

            return BackendRiptrace(
                target_mesh_path=self.config["target_mesh_path"],
                shadow_mesh_path=self.config["shadow_mesh_path"],
                equilibrium_path=self.config["equilibrium_path"],
                target_mesh_type=self.config["target_mesh_type"],
                shadow_mesh_type=self.config["shadow_mesh_type"],
                equilibrium_type=self.config["equilibrium_type"],
                power_loss=self.config["power_loss"],
                lambda_main=self.config["lambda_main"],
                point_omp=self.config["point_omp"],
            )

        if backend_code.lower() == "gyrohit":
            from irflow.backends.flt.gyrohit import BackendGyrohit

            return BackendGyrohit(
                target_mesh_path=self.config["target_mesh_path"],
                shadow_mesh_path=self.config["shadow_mesh_path"],
                equilibrium_path=self.config["equilibrium_path"],
                target_mesh_type=self.config["target_mesh_type"],
                shadow_mesh_type=self.config["shadow_mesh_type"],
                equilibrium_type=self.config["equilibrium_type"],
            )

        raise ValueError(
            f"Unknown field line tracing backend '{backend_code}'. "
            "Supported: 'riptrace', 'gyrohit'."
        )

    def run(self, step_dir: str | Path) -> None:
        """Run the backend in ``step_dir`` (must exist) and store the heat flux."""
        step_dir = Path(step_dir)
        if not step_dir.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {step_dir}")

        self.backend.run(step_dir)
        self.result = {"heat_flux": self.backend.get_heat_flux()}

    def set_irflow_pass_data(self, data: dict[str, Any]) -> None:
        """Field line tracing is the first step, so it takes no upstream data."""
        raise ValueError("FieldLineTracing is the first step and takes no upstream data.")

    def get_irflow_pass_data(self) -> dict[str, np.ndarray]:
        """Return heat flux at target cell centres: ``{"xyz": (M, 3), "heat_flux": (M,)}``."""
        target = self.backend.get_target_mesh_irflow()
        nodes = np.asarray(target["nodes"], dtype=float)
        cells = np.asarray(target["cells_2d"]).astype(int)

        cell_centers = nodes[cells].mean(axis=1)  # (M, 3 nodes, 3 coords) -> (M, 3)
        return {"xyz": cell_centers, "heat_flux": self.backend.get_heat_flux()}
