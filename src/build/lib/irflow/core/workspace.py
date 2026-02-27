# src/irflow/core/workspace.py
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from .step import RunInfo, StepResult, ValidationReport


def _json_default(obj: Any):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class Workspace:
    """
    Owns the on-disk layout for a case.

    Layout (recommended):
      <root>/<case_name>/
        metadata.json
        summary.json
        steps/
          01_flt/
            signature.txt
            validation.json
            run_info.json
            result.json
            backend/<BACKEND_NAME>/...
            artifacts/...
          02_thermal/...
    """

    def __init__(self, root_dir: Path, case_name: str) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.case_name = case_name

    # --------
    # Paths
    # --------

    @property
    def case_dir(self) -> Path:
        return self.root_dir / self.case_name

    @property
    def steps_root(self) -> Path:
        return self.case_dir / "steps"

    def step_dir(self, index: int, step_name: str) -> Path:
        return self.steps_root / f"{index+1:02d}_{step_name}"

    def backend_dir(self, index: int, step_name: str, backend_name: str) -> Path:
        return self.step_dir(index, step_name) / "backend" / backend_name

    def artifacts_dir(self, index: int, step_name: str) -> Path:
        return self.step_dir(index, step_name) / "artifacts"

    # filenames inside step_dir
    def signature_path(self, step_dir: Path) -> Path:
        return step_dir / "signature.txt"

    def validation_path(self, step_dir: Path) -> Path:
        return step_dir / "validation.json"

    def run_info_path(self, step_dir: Path) -> Path:
        return step_dir / "run_info.json"

    def result_path(self, step_dir: Path) -> Path:
        return step_dir / "result.json"

    # ---------------
    # Create / write
    # ---------------

    def create_case_dir(self, exist_ok: bool = True) -> Path:
        self.case_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.steps_root.mkdir(parents=True, exist_ok=True)
        return self.case_dir

    def create_step_dirs(self, index: int, step_name: str, backend_name: str) -> tuple[Path, Path, Path]:
        sdir = self.step_dir(index, step_name)
        bdir = self.backend_dir(index, step_name, backend_name)
        adir = self.artifacts_dir(index, step_name)
        bdir.mkdir(parents=True, exist_ok=True)
        adir.mkdir(parents=True, exist_ok=True)
        return sdir, bdir, adir

    def write_metadata(self, metadata: Mapping[str, Any]) -> Path:
        p = self.case_dir / "metadata.json"
        p.write_text(json.dumps(dict(metadata), indent=2, sort_keys=True, default=_json_default))
        return p

    def write_signature(self, step_dir: Path, signature: str) -> Path:
        p = self.signature_path(step_dir)
        p.write_text(signature.strip() + "\n")
        return p

    def write_validation_report(self, step_dir: Path, report: ValidationReport) -> Path:
        p = self.validation_path(step_dir)
        p.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True, default=_json_default))
        return p

    def write_run_info(self, step_dir: Path, run_info: RunInfo) -> Path:
        p = self.run_info_path(step_dir)
        p.write_text(json.dumps(asdict(run_info), indent=2, sort_keys=True, default=_json_default))
        return p

    def write_step_result(self, step_dir: Path, result: StepResult) -> Path:
        # Store paths as strings for portability
        d = asdict(result)
        d["exchange_h5_path"] = str(result.exchange_h5_path) if result.exchange_h5_path else None
        d["artifacts"] = {k: str(v) for k, v in (result.artifacts or {}).items()}
        if result.run_info is not None:
            d["run_info"] = asdict(result.run_info)
        p = self.result_path(step_dir)
        p.write_text(json.dumps(d, indent=2, sort_keys=True, default=_json_default))
        return p

    def write_summary(self, step_results: list[StepResult]) -> Path:
        p = self.case_dir / "summary.json"
        payload = {
            "case": self.case_name,
            "case_dir": str(self.case_dir),
            "steps": [
                {
                    "step_name": r.step_name,
                    "step_type": r.step_type,
                    "backend": r.backend,
                    "exchange_h5_path": str(r.exchange_h5_path) if r.exchange_h5_path else None,
                    "artifacts": {k: str(v) for k, v in (r.artifacts or {}).items()},
                    "ok": (r.run_info.ok if r.run_info else None),
                }
                for r in step_results
            ],
        }
        p.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
        return p

    # ---------------
    # Read / inspect
    # ---------------

    def inspect_status(self) -> dict[str, Any]:
        """
        Lightweight status summary by scanning step folders for result files.
        """
        out: dict[str, Any] = {
            "case_dir": str(self.case_dir),
            "exists": self.case_dir.exists(),
            "steps": [],
        }
        if not self.steps_root.exists():
            return out

        for step_dir in sorted(self.steps_root.glob("[0-9][0-9]_*")):
            step_info = {
                "step_dir": str(step_dir),
                "has_signature": self.signature_path(step_dir).exists(),
                "has_validation": self.validation_path(step_dir).exists(),
                "has_run_info": self.run_info_path(step_dir).exists(),
                "has_result": self.result_path(step_dir).exists(),
            }
            # pull a tiny bit from result if present
            if step_info["has_result"]:
                try:
                    d = json.loads(self.result_path(step_dir).read_text())
                    step_info["step_name"] = d.get("step_name")
                    step_info["backend"] = d.get("backend")
                    step_info["exchange_h5_path"] = d.get("exchange_h5_path")
                    run_info = d.get("run_info") or {}
                    step_info["ok"] = run_info.get("ok")
                except Exception:
                    step_info["ok"] = None
            out["steps"].append(step_info)
        return out

    # ---------------
    # Cleanup
    # ---------------

    def clean(self, mode: str = "artifacts") -> None:
        """
        mode:
          - "artifacts": delete artifacts dirs only
          - "backend": delete backend dirs only
          - "steps": delete steps entirely
          - "all": delete whole case_dir
        """
        if mode == "all":
            _rm_tree(self.case_dir)
            return
        if not self.steps_root.exists():
            return

        for step_dir in self.steps_root.glob("[0-9][0-9]_*"):
            if mode == "steps":
                _rm_tree(step_dir)
            elif mode == "artifacts":
                _rm_tree(step_dir / "artifacts")
            elif mode == "backend":
                _rm_tree(step_dir / "backend")
            else:
                raise ValueError("mode must be one of: artifacts, backend, steps, all")


def _rm_tree(path: Path) -> None:
    if not path.exists():
        return
    if path.is_file() or path.is_symlink():
        path.unlink()
        return
    # directory
    for p in path.rglob("*"):
        if p.is_file() or p.is_symlink():
            p.unlink()
    for p in sorted(path.rglob("*"), reverse=True):
        if p.is_dir():
            p.rmdir()
    path.rmdir()
