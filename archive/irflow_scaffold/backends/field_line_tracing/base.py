# src/irflow/backends/field_line_tracing/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from irflow.core.step import PreparedCase, StepResult, ValidationReport, RunInfo
from irflow.core.executor import Executor
from irflow.models.resources import Resources
from irflow.models.flt_config import FieldLineTracingConfig

class FLTBackend(ABC):
    name: str  # "Dummy" / "L2G" / ...

    @abstractmethod
    def validate_config(self, config: FieldLineTracingConfig, report: ValidationReport) -> None:
        ...

    @abstractmethod
    def prepare_case(
        self,
        *,
        config: FieldLineTracingConfig,
        upstream_exchange: Optional[Path],
        backend_dir: Path,
        artifacts_dir: Path,
    ) -> PreparedCase:
        ...

    def run(self, prepared: PreparedCase, *, executor: Executor, resources: Resources) -> RunInfo:
        return executor.run(
            prepared.command,
            cwd=prepared.cwd,
            env=prepared.env,
            resources=resources,
            log_path=prepared.log_path,
        )

    @abstractmethod
    def collect_outputs(
        self,
        *,
        config: FieldLineTracingConfig,
        backend_dir: Path,
        artifacts_dir: Path,
        run_info: RunInfo,
    ) -> StepResult:
        ...
