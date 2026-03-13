# src/irflow/steps/field_line_tracing.py
from __future__ import annotations
from pathlib import Path
import json
import hashlib

#from BackendRiptrace import BackendRiptrace
#from irflow.core.step import Step, PreparedCase, StepResult, ValidationReport, RunInfo
#from irflow.core.registry import BackendRegistry
#from irflow.core.executor import Executor
#from irflow.models.flt_config import FieldLineTracingConfig
#from irflow.models.resources import Resources
#from irflow.models.artifacts import ArtifactOptions

class ThermalModel():
    step_type = "flt"

    def __init__(self, name='tm_case', enabled=True,
                 backend_code="elmerfem", config=None,
                 resources=None, root_dir=None):
        self.name = name #.validate_light()
        self.enabled = enabled #.validate()
        self.config = config #.validate_light()
        self.backend_code = backend_code #.validate()
        self.resources = resources # or Resources()
        self.root_dir = root_dir
        #self.artifacts = artifacts or ArtifactOptions()
        #self.validate_config_light()

        self.nodes = None
        self.cells_3d = None
        self.cells_2d = None
        self.groups_3d = None
        self.groups_2d = None
        self.material_properties = None
        self.boundary_conditions = None

        self.numerical_parameters = None

        self.result = None

        self.backend = self.get_backend(self.backend_code)

        #self.data = self.backend.get_irflow_data()

    def set_irflow_pass_data(self, data_to_pass):
        self.backend.get_irflow_pass_data(data_to_pass)

    def get_backend(self, backend_code):
        if backend_code.lower() == "elmerfem":
            from BackendElmerFEM import BackendElmerFEM
            mesh_file = self.config["mesh_file"]
            mesh_type = self.config["mesh_type"]
            sif_file = self.config["sif_file"]
            plasma_bc_id = self.config["plasma_bc_id"]
            time_stepping = self.config["time_stepping"]
            return BackendElmerFEM(mesh_file, mesh_type, sif_file,
                                   plasma_bc_id, self.root_dir, time_stepping)

        elif backend_code.lower() == "openfoam":
            from BackendOpenFOAM import BackendOpenFOAM
            mesh_file = self.config["mesh_file"]
            mesh_type = self.config["mesh_type"]
            plasma_bc_name = self.config["plasma_bc_name"]
            time_stepping = self.config["time_stepping"]
            openfoam_bashrc_dir = self.config["openfoam_bashrc_dir"]
            initial_temperature = self.config.get("initial_temperature", None)
            boundary_conditions = self.config.get("boundary_conditions", None)
            return BackendOpenFOAM(mesh_file, mesh_type, plasma_bc_name,
                                   self.root_dir, time_stepping,
                                   openfoam_bashrc_dir,
                                   initial_temperature=initial_temperature,
                                   boundary_conditions=boundary_conditions)

    #def get_flt_data(self, ):
    #    return self.backend.get_irflow_data()

    def run(self, irflow_case_dir):
        path = Path(irflow_case_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {path}")

        path = Path(irflow_case_dir, "ThermalModel")
        if not path.is_dir():
            path.mkdir(parents=True)

        #path = Path(irflow_case_dir, "FieldLineTracing")
        #if not path.is_dir():
        #    path.mkdir(parents=True)

        # Prepare case (if needed)
        #self.backend.prepare(path)

        # Run case
        self.backend.run(path)

        # Get result (heat flux on target grid) from backend
        temperature = self.backend.get_temperature()
        self.result = {"temperature": temperature}

    def get_irflow_pass_data(self, ):
        # Pass only heat flux coodinates of heat flux
        __import__("pdb").set_trace()

        nodes = self.backend.get_irflow_data()["target"]["nodes"]
        cells = self.backend.get_irflow_data()["target"]["cells_2d"]
        heat_flux = self.backend.get_heat_flux()

        # Gather node coordinates for each cell
        cell_nodes = nodes[cells] # shape (Ncells, 3 nodes, 3 coords) # Compute centers by averaging over the 3 nodes
        cell_centers = cell_nodes.mean(axis=1) # shape (Ncells, 3)
        irflow_pass_data = {"xyz": cell_centers, "heat_flux": heat_flux}
        
        return irflow_pass_data
    




    #def validate_config_light(self, ):
    #    if self.backend.lower is "riptrace":
    '''
    def validate(self, upstream_exchange: Path | None, *, registry: BackendRegistry) -> ValidationReport:
        report = ValidationReport(step=self.name)

        # Generic check: surface_mesh must be specified (existence check optional for dummy)
        if self.config.surface_mesh is None:
            report.error("surface_mesh is required")

        # Backend-specific
        backend = registry.get(self.step_type, self.backend)
        backend.validate_config(self.config, report)

        report.raise_if_errors()
        return report

    def signature(self, upstream_exchange: Path | None) -> str:
        # Minimal deterministic signature: config dict + upstream path string
        payload = {
            "backend": self.backend,
            "surface_mesh": str(self.config.surface_mesh),
            "equilibrium": str(self.config.equilibrium) if self.config.equilibrium else None,
            "bfield_3d": str(self.config.bfield_3d) if self.config.bfield_3d else None,
            "output_location": self.config.output_location,
            "output_fields": list(self.config.output_fields),
            "backend_options": self.config.backend_options,
            "upstream_exchange": str(upstream_exchange) if upstream_exchange else None,
        }
        s = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(s).hexdigest()

    def prepare(
        self,
        *,
        upstream_exchange: Path | None,
        backend_dir: Path,
        artifacts_dir: Path,
        registry: BackendRegistry,
    ) -> PreparedCase:
        backend = registry.get(self.step_type, self.backend)
        return backend.prepare_case(
            config=self.config,
            upstream_exchange=upstream_exchange,
            backend_dir=backend_dir,
            artifacts_dir=artifacts_dir,
        )

    def run(self, prepared: PreparedCase, *, executor: Executor) -> RunInfo:
        # Let backend run (useful later for multi-command backends)
        # For now, call executor directly or backend.run — either works.
        return executor.run(
            prepared.command,
            cwd=prepared.cwd,
            env=prepared.env,
            resources=self.resources,
            log_path=prepared.log_path,
        )

    def collect(
        self,
        *,
        upstream_exchange: Path | None,
        backend_dir: Path,
        artifacts_dir: Path,
        run_info: RunInfo,
        registry: BackendRegistry,
    ) -> StepResult:
        backend = registry.get(self.step_type, self.backend)
        return backend.collect_outputs(
            config=self.config,
            backend_dir=backend_dir,
            artifacts_dir=artifacts_dir,
            run_info=run_info,
        )
    '''
