# src/irflow/core/executor.py
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

from .step import RunInfo
from irflow.models.resources import Resources  # if you don't have this yet, replace with Any


@dataclass
class Executor:
    """Base class. LocalExecutor is the first concrete implementation."""
    name: str = "executor"


@dataclass
class LocalExecutor(Executor):
    """
    Runs commands locally via subprocess.

    Notes:
      - Resources are applied lightly (OMP_NUM_THREADS, env); no scheduler.
      - stdout/stderr are merged into one log unless you provide separate paths.
    """
    name: str = "local"
    shell: bool = False
    extra_env: dict[str, str] = field(default_factory=dict)

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Optional[Mapping[str, str]] = None,
        resources: Optional[Resources] = None,
        log_path: Optional[Path] = None,
        timeout_s: Optional[int] = None,
    ) -> RunInfo:
        cwd = Path(cwd).expanduser().resolve()
        cwd.mkdir(parents=True, exist_ok=True)

        merged_env = os.environ.copy()
        merged_env.update(self.extra_env)
        if env:
            merged_env.update({str(k): str(v) for k, v in env.items()})

        # Apply minimal resource hints
        if resources is not None:
            if getattr(resources, "omp_threads", None):
                merged_env["OMP_NUM_THREADS"] = str(resources.omp_threads)

        stdout_path = None
        stderr_path = None

        # Logging
        if log_path is not None:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            # merge stdout+stderr into one file for simplicity
            log_f = open(log_path, "w", buffering=1)
            stdout = log_f
            stderr = subprocess.STDOUT
            stdout_path = log_path
        else:
            stdout = subprocess.PIPE
            stderr = subprocess.PIPE

        t0 = time.time()
        try:
            proc = subprocess.run(
                list(command),
                cwd=str(cwd),
                env=merged_env,
                stdout=stdout,
                stderr=stderr,
                text=True if log_path is None else False,
                timeout=timeout_s,
                check=False,
                shell=self.shell,
            )
            runtime = time.time() - t0

            if log_path is None:
                # If no log_path, capture and optionally store to default files later
                message = None
                if proc.stdout:
                    message = proc.stdout[-2000:]  # last 2k chars
            else:
                message = None

            ok = (proc.returncode == 0)
            return RunInfo(
                ok=ok,
                exit_code=int(proc.returncode),
                runtime_s=runtime,
                message=message,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
        except subprocess.TimeoutExpired as e:
            runtime = time.time() - t0
            return RunInfo(
                ok=False,
                exit_code=124,
                runtime_s=runtime,
                message=f"Timeout after {timeout_s}s",
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                extra={"exception": str(e)},
            )
        except Exception as e:
            runtime = time.time() - t0
            return RunInfo(
                ok=False,
                exit_code=1,
                runtime_s=runtime,
                message=str(e),
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                extra={"exception": repr(e)},
            )
        finally:
            if log_path is not None:
                try:
                    log_f.close()
                except Exception:
                    pass
