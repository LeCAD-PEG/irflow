"""Simulation step 2: thermal model, giving the temperature of the plasma-facing surface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from irflow.backends.base import ThermalModelBackend
from irflow.config import check_config, check_time_stepping
from irflow.data import CameraImages, HeatFluxField, TemperatureField
from irflow.io import load_heat_flux


class ThermalModel:
    """Thermal model step. The backend code is chosen with ``backend_code``.

    By default the heat flux comes from the previous (field line tracing) step. To run the
    thermal model on its own, give ``input_data``: a :class:`HeatFluxField` or the path of a
    heat flux file saved by irflow (e.g. ``FieldLineTracing/heat_flux.vtp``). For other
    codes' files use :func:`irflow.io.load_heat_flux` with the right ``array_name``.

    Optional config key ``mapping_tolerance`` [m]: largest allowed distance between the
    plasma surface and the heat flux points (default: twice the longest edge of either mesh).
    """

    dir_name = "ThermalModel"

    def __init__(
        self,
        name: str = "tm_case",
        enabled: bool = True,
        backend_code: str = "elmerfem",
        config: dict[str, Any] | None = None,
        resources: Any = None,
        input_data: HeatFluxField | str | Path | None = None,
    ) -> None:
        self.name = name
        self.enabled = enabled
        self.backend_code = backend_code
        self.config = config or {}
        self.resources = resources
        self.result: TemperatureField | None = None

        if isinstance(input_data, (str, Path)):
            input_data = load_heat_flux(input_data)
        self.heat_flux: HeatFluxField | None = input_data
        self.has_own_input = input_data is not None

        self.backend = self.get_backend(backend_code)

    def get_backend(self, backend_code: str) -> ThermalModelBackend:
        """Create the backend for ``backend_code`` from the step config."""
        # Backends are imported here so that only the chosen code has to be installed.
        where = f"ThermalModel config (backend '{backend_code}')"
        common_optional = ["time_stepping", "mapping_tolerance"]
        time_stepping = self.config.get("time_stepping", {"type": "steady"})
        check_time_stepping(time_stepping, where=f"{where}, time_stepping")

        if backend_code.lower() == "elmerfem":
            from irflow.backends.tm.elmerfem import BackendElmerFEM

            check_config(
                self.config,
                required=["mesh_file", "mesh_type", "sif_file", "plasma_bc_id"],
                optional=common_optional,
                where=where,
            )
            return BackendElmerFEM(
                mesh_file=self.config["mesh_file"],
                mesh_type=self.config["mesh_type"],
                sif_file=self.config["sif_file"],
                plasma_bc_id=self.config["plasma_bc_id"],
                time_stepping=time_stepping,
            )

        if backend_code.lower() == "openfoam":
            from irflow.backends.tm.openfoam.backend import BackendOpenFOAM

            check_config(
                self.config,
                required=["mesh_file", "mesh_type", "plasma_bc_name"],
                optional=common_optional
                + [
                    "openfoam_bashrc",
                    "initial_temperature",
                    "boundary_conditions",
                    "material_properties",
                ],
                where=where,
            )
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

        if backend_code.lower() == "simple1d":
            from irflow.backends.tm.simple1d import BackendSimple1D

            check_config(
                self.config,
                required=["thickness", "material_properties", "back_boundary"],
                optional=common_optional
                + [
                    "initial_temperature",
                    "n_points",
                    "first_cell_size",
                    "unused_node_temperature",
                ],
                where=where,
            )
            return BackendSimple1D(self.config)

        raise ValueError(
            f"Unknown thermal model backend '{backend_code}'. "
            "Supported: 'elmerfem', 'openfoam', 'simple1d'."
        )

    def set_irflow_pass_data(
        self, data: HeatFluxField | TemperatureField | CameraImages
    ) -> None:
        """Receive the heat flux from the previous step."""
        if not isinstance(data, HeatFluxField):
            raise TypeError(f"ThermalModel needs a HeatFluxField, got {type(data).__name__}")
        self.heat_flux = data

    def run(self, step_dir: str | Path) -> None:
        """Run the backend in ``step_dir`` (must exist) and store the surface temperature."""
        step_dir = Path(step_dir)
        if not step_dir.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {step_dir}")
        if self.heat_flux is None:
            raise RuntimeError(
                "ThermalModel has no heat flux: run a FieldLineTracing step before it "
                "or give input_data."
            )

        self.backend.run(step_dir, self.heat_flux, self.config.get("mapping_tolerance"))
        self.result = self.backend.get_result()

    def get_irflow_pass_data(self) -> TemperatureField:
        """Plasma-facing surface temperature, for the optical step."""
        if self.result is None:
            raise RuntimeError("ThermalModel has no result yet: call run() first.")
        return self.result
