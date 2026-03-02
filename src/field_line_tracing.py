# src/irflow/steps/field_line_tracing.py
from __future__ import annotations
from pathlib import Path
import json
import hashlib

from irflow.core.step import Step, PreparedCase, StepResult, ValidationReport, RunInfo
from irflow.core.registry import BackendRegistry
from irflow.core.executor import Executor
from irflow.models.flt_config import FieldLineTracingConfig
from irflow.models.resources import Resources
from irflow.models.artifacts import ArtifactOptions

class FieldLineTracing(Step):
    step_type = "flt"

    def __init__(self, name='flt_case', enabled=True,
                 backend_code="riptrace", config=None, resources=None):
        self.name = name #.validate_light()
        self.enabled = enabled #.validate()
        self.config = config #.validate_light()
        self.backend_code = backend_code #.validate()
        self.resources = resources or Resources()
        #self.artifacts = artifacts or ArtifactOptions()

        self.validate_config_light()

        self.shadow = None
        self.target = None
        self.equilibrium = None
        self.numerical_parameters = None

        self.backend = self.get_backend(self.backend_code)
        self.data = self.backend.get_irflow_data()

    def get_backend(self, backend_code):
        if backend_code.lower() is "riptrace":
            from irflow.backends import BackendRiptrace
            target_mesh_path = self.config["target_mesh_path"]
            shadow_mesh_path = self.config["shadow_mesh_path"]
            equilibrium_path = self.config["equilibrium_path"]
            target_mesh_type = self.config["target_mesh_type"]
            shadow_mesh_type = self.config["shadow_mesh_type"]
            equilibrium_type = self.config["equilibrium_type"]
            power_loss = self.config["power_loss"]
            lambda_main = self.config["lambda_main"]
            point_omp = self.config["point_omp"]

            return BackendRiptrace(target_mesh_path, shadow_mesh_path,
                                   equilibrium_path,
                                   target_mesh_type,
                                   shadow_mesh_type, equilibrium_type,
                                   power_loss, lambda_main,
                                   point_omp)
            

    def get_flt_data(self, ):
        return self.backend.get_irflow_data()

    def run_flt_case(self, irflow_case_dir):
        path = Path(irflow_case_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {path}")

        path = Path(irflow_case_dir, "FieldLineTracing")
        if not path.is_dir():
            path.mkdir(parents=True)

        path = Path(irflow_case_dir, "FieldLineTracing", self.backend_code)
        if not path.is_dir():
            path.mkdir(parents=True)

        # Prepare case (if needed)
        self.backend.prepare(path)

        # Run case
        self.backend.run()

        # Get result (heat flux on target grid) from backend
        result = self.backend.get_irflow_result()

        
                
        
        

    #def validate_config_light(self, ):
    #    if self.backend.lower is "riptrace":
            
            
        


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
