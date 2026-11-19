# src/irflow/models/flt_config.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

BackendName = Literal["Dummy", "L2G", "Riptrace", "SMARDDA"]

@dataclass(frozen=True)
class FieldLineTracingConfig:
    backend: BackendName
    surface_mesh: str | Path

    # L2G-like
    equilibrium: str | Path | None = None
    # Riptrace-like
    bfield_3d: str | Path | None = None
    field_format: str | None = None

    # minimal numerical options
    output_location: Literal["cells", "nodes"] = "cells"
    output_fields: tuple[str, ...] = ("q",)

    backend_options: dict[str, Any] = field(default_factory=dict)

    def validate_light(self) -> None:
        if not self.backend:
            raise ValueError("backend must be set")
        if not self.output_fields:
            raise ValueError("output_fields cannot be empty")
        if self.output_location not in ("cells", "nodes"):
            raise ValueError("output_location must be 'cells' or 'nodes'")
