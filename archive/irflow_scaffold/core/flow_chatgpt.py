# src/irflow/core/flow.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from irflow.core.workspace import Workspace
from irflow.core.registry import BackendRegistry
from irflow.core.executor import Executor
from irflow.core.step import Step, StepResult

@dataclass
class FlowResult:
    case_dir: Path
    step_results: list[StepResult]

class IRFlow:
    def __init__(self, *, case_name: str, root_dir: str | Path, executor: Executor, registry: BackendRegistry, metadata: dict | None = None):
        self.case_name = case_name
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.executor = executor
        self.registry = registry
        self.metadata = metadata or {}
        self.steps: list[Step] = []
        self.workspace = Workspace(self.root_dir, self.case_name)

    def add_step(self, step: Step) -> None:
        self.steps.append(step)

    def run(self) -> FlowResult:
        self.workspace.create_case_dir()
        self.workspace.write_metadata(self.metadata)

        upstream_exchange: Optional[Path] = None
        results: list[StepResult] = []

        for i, step in enumerate(self.steps):
            if not step.enabled:
                continue

            step_dir, backend_dir, artifacts_dir = self.workspace.create_step_dirs(i, step.name, step.backend)

            report = step.validate(upstream_exchange, registry=self.registry)
            self.workspace.write_validation_report(step_dir, report)

            sig = step.signature(upstream_exchange)
            self.workspace.write_signature(step_dir, sig)

            prepared = step.prepare(
                upstream_exchange=upstream_exchange,
                backend_dir=backend_dir,
                artifacts_dir=artifacts_dir,
                registry=self.registry,
            )
            run_info = step.run(prepared, executor=self.executor)
            self.workspace.write_run_info(step_dir, run_info)

            result = step.collect(
                upstream_exchange=upstream_exchange,
                backend_dir=backend_dir,
                artifacts_dir=artifacts_dir,
                run_info=run_info,
                registry=self.registry,
            )
            self.workspace.write_step_result(step_dir, result)

            upstream_exchange = result.exchange_h5_path
            results.append(result)

        self.workspace.write_summary(results)
        return FlowResult(case_dir=self.workspace.case_dir, step_results=results)
