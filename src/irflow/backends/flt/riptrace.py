"""Field line tracing backend for Riptrace (in-house code, runs in-process via riptrace_py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from riptrace_py.riptrace_case import RipTraceCase
from riptrace_py.riptrace_mesh import Shadow, Target

from irflow.backends.base import FieldLineTracingBackend
from irflow.config import check_config
from irflow.data import HeatFluxField, SurfaceMesh


class BackendRiptrace(FieldLineTracingBackend):
    """Connector between the irflow field line tracing step and Riptrace."""

    def __init__(
        self,
        target_mesh_path: str,
        shadow_mesh_path: str,
        equilibrium_path: dict[str, str],
        target_mesh_type: str,
        shadow_mesh_type: str,
        equilibrium_type: str,
        power_loss: float,
        lambda_main: float,
        point_omp: list[float],
        numerical_parameters: Any = None,
        hpc: dict[str, Any] | None = None,
    ) -> None:
        # input parameters
        self.target_mesh_path = target_mesh_path
        self.target_mesh_type = target_mesh_type
        self.shadow_mesh_path = shadow_mesh_path
        self.shadow_mesh_type = shadow_mesh_type
        self.equilibrium_path = equilibrium_path
        self.equilibrium_type = equilibrium_type

        self.power_loss = power_loss  # Power in the scrape-off layer [W]
        self.lambda_main = lambda_main  # main decay length [m]
        self.point_omp = point_omp  # Location of separatrix/LCFS at outer midplane R, Z [m]
        self.numerical_params = numerical_parameters  # numerical parameters for FLT
        # Execution resources, e.g. {"backend": "mpi", "omp_threads": 8} for OpenMP;
        # default: Riptrace's serial CPU backend
        self.hpc = hpc or {}
        check_config(self.hpc, required=[], optional=["backend", "omp_threads"], where="hpc")

        # backend data format (private)
        self.__target_riptrace: Target | None = None
        self.__shadow_riptrace: Shadow | None = None
        self.__magnetic_field_riptrace: Any = None
        self.__magnetic_flux_riptrace: Any = None
        self.case_riptrace: RipTraceCase | None = None

        # irflow data format (to be passed back to FLT class)
        self.target_irflow: dict[str, Any] | None = None
        self.shadow_irflow: dict[str, Any] | None = None
        self.equilibrium_irflow: dict[str, Any] | None = None

    def get_target_mesh_riptrace(self) -> Target:
        self.__target_riptrace = Target(name="irflow_target")
        if self.target_mesh_type.lower() == "dat":
            self.__target_riptrace.load_dat(self.target_mesh_path)
        elif self.target_mesh_type.lower() == "vtk":
            self.__target_riptrace.load_vtk_mesh(self.target_mesh_path)
        else:
            raise ValueError(
                f"Unknown target mesh type '{self.target_mesh_type}'. "
                "Supported types: 'vtk' and 'dat'."
            )
        return self.__target_riptrace

    def get_target_mesh_irflow(self) -> dict[str, Any]:
        target = self.__target_riptrace or self.get_target_mesh_riptrace()
        self.target_irflow = {"nodes": target.nodes, "cells_2d": target.triangles}
        return self.target_irflow

    def get_shadow_mesh_riptrace(self) -> Shadow:
        self.__shadow_riptrace = Shadow(name="irflow_shadow")
        if self.shadow_mesh_type.lower() == "vtk":
            self.__shadow_riptrace.load_vtk_mesh(self.shadow_mesh_path)
        else:
            raise ValueError(
                f"Unknown shadow mesh type '{self.shadow_mesh_type}'. "
                "Supported type: 'vtk'."
            )
        return self.__shadow_riptrace

    def get_shadow_mesh_irflow(self) -> dict[str, Any]:
        shadow = self.__shadow_riptrace or self.get_shadow_mesh_riptrace()
        self.shadow_irflow = {"nodes": shadow.nodes, "cells_2d": shadow.triangles}
        return self.shadow_irflow

    def get_equilibrium(self) -> dict[str, Any]:
        if self.equilibrium_type.lower() == "pfcflux":
            from riptrace_py.io_reader import read_pfcflux_data

            self.__magnetic_field_riptrace, self.__magnetic_flux_riptrace = read_pfcflux_data(
                self.equilibrium_path.get("txt")
            )

        elif self.equilibrium_type.lower() == "smardda":
            from riptrace_py.io_reader import read_magtfm_data

            self.__magnetic_field_riptrace, self.__magnetic_flux_riptrace = read_magtfm_data(
                self.equilibrium_path.get("fiesta"), self.equilibrium_path.get("ov5")
            )

        elif self.equilibrium_type.lower() == "energopul":
            raise NotImplementedError(
                "Equilibrium type 'energopul' is recognized but not implemented yet."
            )
        else:
            raise ValueError(
                f"Unknown equilibrium type '{self.equilibrium_type}'. "
                "Supported types: 'pfcflux' and 'smardda'."
            )

        return {
            "phi_1d": self.__magnetic_field_riptrace.phi_1d,
            "z_1d": self.__magnetic_field_riptrace.z_1d,
            "r_1d": self.__magnetic_field_riptrace.r_1d,
            "bphi_1d": self.__magnetic_field_riptrace.bphi_1d,
            "bz_1d": self.__magnetic_field_riptrace.bz_1d,
            "br_1d": self.__magnetic_field_riptrace.br_1d,
            "psi_1d": self.__magnetic_flux_riptrace.psi_1d,
        }

    def get_numerical_params(self) -> dict[str, Any]:
        if self.numerical_params is None:
            from riptrace_py.riptrace_case import RipTraceNumericalParams

            self.numerical_params = RipTraceNumericalParams()

        return {
            "h_initial": self.numerical_params.h_initial,
            "tolerance": self.numerical_params.tolerance,
            "h_min": self.numerical_params.h_min,
            "h_max": self.numerical_params.h_max,
            "max_points": self.numerical_params.max_points,
            "max_length": self.numerical_params.max_length,
            "is_shadowed": self.numerical_params.is_shadowed,
            "offset": self.numerical_params.offset,
            "kdtree_max_depth": self.numerical_params.kdtree_max_depth,
            "debug_vtk": self.numerical_params.debug_vtk,
        }

    def get_irflow_data(self) -> dict[str, Any]:
        """Return all FLT inputs in irflow format (loads meshes and equilibrium)."""
        equilibrium = self.get_equilibrium()
        equilibrium["power_loss"] = self.power_loss
        equilibrium["lambda_main"] = self.lambda_main
        equilibrium["point_omp"] = self.point_omp

        return {
            "target": self.get_target_mesh_irflow(),
            "shadow": self.get_shadow_mesh_irflow(),
            "equilibrium": equilibrium,
            "num_params": self.get_numerical_params(),
        }

    def run(self, case_dir: Path) -> None:
        """Load inputs, trace field lines and compute power deposition on the target."""
        self.get_target_mesh_riptrace()
        self.get_shadow_mesh_riptrace()
        self.get_equilibrium()

        riptrace_case_dir = Path(case_dir) / "riptrace_case"
        self.case_riptrace = RipTraceCase(
            "smardda_case",
            target=self.__target_riptrace,
            shadow=self.__shadow_riptrace,
            magnetic_field=self.__magnetic_field_riptrace,
            magnetic_flux=self.__magnetic_flux_riptrace,
            case_path=riptrace_case_dir,
            hpc_params=self.get_hpc_params(),
        )

        self.case_riptrace.compute_wetted_area()
        self.case_riptrace.compute_power_deposition(
            self.power_loss, self.lambda_main, 1 / 2, self.point_omp
        )

    def get_hpc_params(self) -> Any:
        """Riptrace execution resources (in this process) from the ``hpc`` config."""
        from riptrace_py.hpc import RipTraceHPCParams

        hpc_params = RipTraceHPCParams(backend=self.hpc.get("backend", "cpu"), mode="inprocess")
        if "omp_threads" in self.hpc:
            hpc_params.omp_threads = self.hpc["omp_threads"]
        return hpc_params

    def get_result(self) -> HeatFluxField:
        """Heat flux on each target triangle [W/m^2]."""
        if self.case_riptrace is None:
            raise RuntimeError("Riptrace has not been run yet.")
        target = self.case_riptrace.target
        mesh = SurfaceMesh(nodes=target.nodes, triangles=target.triangles)
        return HeatFluxField(mesh=mesh, heat_flux=np.ravel(target.Q))
