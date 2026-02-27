# src/irflow/core/step.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


# -----------------------------
# Errors / validation reporting
# -----------------------------

class IRFlowError(Exception):
    """Base error for IRFlow."""


class ConfigError(IRFlowError):
    """Raised when configuration is invalid."""


class RunError(IRFlowError):
    """Raised when execution fails."""


@dataclass
class ValidationReport:
    """Collects validation errors/warnings without immediately raising."""
    step: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_info(self, msg: str) -> None:
        self.info.append(msg)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def raise_if_errors(self) -> None:
        if self.errors:
            joined = "\n".join(f"- {e}" for e in self.errors)
            raise ConfigError(f"Validation failed for step '{self.step}':\n{joined}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# -----------------------------
# Step execution core types
# -----------------------------

@dataclass(frozen=True)
class PreparedCase:
    """
    Output of Step.prepare() / Backend.prepare_case().

    `command` should be a list suitable for subprocess, e.g. ["l2g", "-i", "input.in"].
    """
    command: Sequence[str]
    cwd: Path
    env: Mapping[str, str] = field(default_factory=dict)
    log_path: Optional[Path] = None
    expected_outputs: Sequence[Path] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunInfo:
    """Result of actually running the backend command(s)."""
    ok: bool
    exit_code: int = 0
    runtime_s: float | None = None
    job_id: str | None = None  # for schedulers
    message: str | None = None
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepResult:
    """
    Standardized result returned by Step.collect().

    `exchange_h5_path` is the handover artifact to the next step (or None if final).
    """
    step_name: str
    step_type: str
    backend: str
    exchange_h5_path: Path | None
    artifacts: Mapping[str, Path] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    run_info: RunInfo | None = None
    signature: str | None = None  # cache key if you use caching


# -----------------------------
# Base Step interface
# -----------------------------

class Step(ABC):
    """
    Backend-agnostic step controller interface used by IRFlow.

    Concrete steps (FieldLineTracing/ThermalModelling/CameraSimulator) implement these
    and delegate backend-specific work to adapters obtained from a registry.
    """

    # e.g. "flt", "thermal", "camera"
    step_type: str

    # short name instance label, e.g. "flt"
    name: str

    # backend identifier, e.g. "L2G"
    backend: str

    enabled: bool

    @abstractmethod
    def validate(self, upstream_exchange: Optional[Path], *, registry: Any) -> ValidationReport:
        """
        Validate configuration for this step.
        Should perform generic checks and call backend.validate_config(...).
        """
        raise NotImplementedError

    @abstractmethod
    def signature(self, upstream_exchange: Optional[Path]) -> str:
        """
        Return a stable signature used for caching (hash of config + upstream inputs).
        Must be deterministic across runs.
        """
        raise NotImplementedError

    @abstractmethod
    def prepare(
        self,
        *,
        upstream_exchange: Optional[Path],
        backend_dir: Path,
        artifacts_dir: Path,
        registry: Any,
    ) -> PreparedCase:
        """
        Write backend case inputs into backend_dir and return a PreparedCase
        describing how to run it.
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, prepared: PreparedCase, *, executor: Any) -> RunInfo:
        """Execute the prepared case using an Executor (local/slurm/etc.)."""
        raise NotImplementedError

    @abstractmethod
    def collect(
        self,
        *,
        upstream_exchange: Optional[Path],
        backend_dir: Path,
        artifacts_dir: Path,
        run_info: RunInfo,
        registry: Any,
    ) -> StepResult:
        """
        Parse backend outputs and write standardized artifacts (exchange H5, VTK, images).
        Return StepResult.
        """
        raise NotImplementedError
