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

    def __init__(
        self,
        config: FieldLineTracingConfig,
        *,
        resources: Resources | None = None,
        artifacts: ArtifactOptions | None = None,
        name: str = "flt",
        enabled: bool = True,
    ) -> None:
        self.name = name
        self.enabled = enabled
        self.config = config
        self.backend = str(config.backend)
        self.resources = resources or Resources()
        self.artifacts = artifacts or ArtifactOptions()

        self.config.validate_light()
        self.resources.validate()
        self.artifacts.validate()

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
