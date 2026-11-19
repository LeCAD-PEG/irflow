"""Simulation step 2: thermal model, giving the temperature of the plasma-facing surface."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from irflow.backends.tm.elmerfem import BackendElmerFEM
    from irflow.backends.tm.openfoam.backend import BackendOpenFOAM


class ThermalModel:
    """Thermal model step. The backend code is chosen with ``backend_code``."""

    dir_name = "ThermalModel"

    def __init__(
        self,
        name: str = "tm_case",
        enabled: bool = True,
        backend_code: str = "elmerfem",
        config: dict[str, Any] | None = None,
        resources: Any = None,
    ) -> None:
        self.name = name
        self.enabled = enabled
        self.backend_code = backend_code
        self.config = config or {}
        self.resources = resources
        self.result: dict[str, np.ndarray] | None = None

        self.backend = self.get_backend(backend_code)

    def get_backend(self, backend_code: str) -> BackendElmerFEM | BackendOpenFOAM:
        """Create the backend for ``backend_code`` from the step config."""
        # Backends are imported here so that only the chosen code has to be installed.
        time_stepping = self.config.get("time_stepping", {"type": "steady"})

        if backend_code.lower() == "elmerfem":
            from irflow.backends.tm.elmerfem import BackendElmerFEM

            return BackendElmerFEM(
                mesh_file=self.config["mesh_file"],
                mesh_type=self.config["mesh_type"],
                sif_file=self.config["sif_file"],
                plasma_bc_id=self.config["plasma_bc_id"],
                time_stepping=time_stepping,
            )

        if backend_code.lower() == "openfoam":
            from irflow.backends.tm.openfoam.backend import BackendOpenFOAM

            return BackendOpenFOAM(
                mesh_file=self.config["mesh_file"],
                mesh_type=self.config["mesh_type"],
                plasma_bc_name=self.config["plasma_bc_name"],
                time_stepping=time_stepping,
                openfoam_bashrc=self.config.get("openfoam_bashrc"),
                initial_temperature=self.config.get("initial_temperature", 300.0),
                boundary_conditions=self.config.get("boundary_conditions"),
                material_properties=self.config.get("material_properties"),
            )

        raise ValueError(
            f"Unknown thermal model backend '{backend_code}'. Supported: 'elmerfem', 'openfoam'."
        )

    def set_irflow_pass_data(self, data: dict[str, Any]) -> None:
        """Receive heat flux from the previous step: ``{"xyz": (M, 3), "heat_flux": (M,)}``."""
        self.backend.set_irflow_pass_data(data)

    def run(self, step_dir: str | Path) -> None:
        """Run the backend in ``step_dir`` (must exist) and store the surface temperature."""
        step_dir = Path(step_dir)
        if not step_dir.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {step_dir}")

        self.backend.run(step_dir)
        self.result = self.backend.get_temperature()

    def get_irflow_pass_data(self) -> dict[str, np.ndarray]:
        """Return the plasma-facing surface temperature for the optical step.

        Keys: ``"nodes"`` (N, 3), ``"cells_2d"`` (M, 3), ``"times"`` (Nt,) and
        ``"temperature"`` (Nt, N) at the nodes. A steady run has one time.
        """
        if self.result is None:
            raise RuntimeError("ThermalModel has no result yet: call run() first.")
        return self.result
