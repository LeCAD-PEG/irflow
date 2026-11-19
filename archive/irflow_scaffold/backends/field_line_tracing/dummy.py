# src/irflow/backends/field_line_tracing/dummy.py
from __future__ import annotations
from pathlib import Path
import numpy as np

from .base import FLTBackend
from irflow.core.step import PreparedCase, StepResult, ValidationReport, RunInfo
from irflow.core.exchange import Exchange, Mesh, write_exchange_h5, EXCHANGE_SCHEMA_VERSION
from irflow.models.flt_config import FieldLineTracingConfig

class DummyFLTBackend(FLTBackend):
    name = "Dummy"

    def validate_config(self, config: FieldLineTracingConfig, report: ValidationReport) -> None:
        # dummy backend only needs a surface_mesh path string (can be non-existent for now)
        if config.surface_mesh is None:
            report.error("surface_mesh must be provided")

    def prepare_case(self, *, config, upstream_exchange, backend_dir: Path, artifacts_dir: Path) -> PreparedCase:
        backend_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # No real command needed; use a harmless command that always succeeds.
        cmd = ["python", "-c", "print('Dummy FLT: no-op run')"]
        return PreparedCase(
            command=cmd,
            cwd=backend_dir,
            env={},
            log_path=artifacts_dir / "run.log",
        )

    def collect_outputs(self, *, config, backend_dir: Path, artifacts_dir: Path, run_info: RunInfo) -> StepResult:
        # Create a tiny triangular mesh (2 triangles forming a square)
        nodes = np.array(
            [[0.0, 0.0, 0.0],
             [1.0, 0.0, 0.0],
             [1.0, 1.0, 0.0],
             [0.0, 1.0, 0.0]], dtype=np.float64
        )
        triangles = np.array([[0, 1, 2],
                              [0, 2, 3]], dtype=np.int64)

        # Fake heat flux on cells (2 cells)
        q = np.array([1.0e6, 2.0e6], dtype=np.float64)  # W/m^2

        ex = Exchange(
            schema_version=EXCHANGE_SCHEMA_VERSION,
            step_type="flt",
            backend=self.name,
            mesh=Mesh(nodes=nodes, triangles=triangles),
            fields={"q": q},
            field_location={"q": "cells"},
            units={"q": "W/m^2"},
            metadata={"note": "dummy exchange"},
            provenance={"backend": self.name},
        )

        exchange_path = artifacts_dir / "flt_exchange.h5"
        write_exchange_h5(
            ex,
            exchange_path,
            compression="gzip",
            compression_level=4,
            float_dtype="float32",
            overwrite=True,
        )

        return StepResult(
            step_name="flt",
            step_type="flt",
            backend=self.name,
            exchange_h5_path=exchange_path,
            artifacts={"log": artifacts_dir / "run.log"},
            provenance={"backend": self.name},
            run_info=run_info,
        )
