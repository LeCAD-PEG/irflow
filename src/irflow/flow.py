"""Top-level workflow: runs the simulation steps in order and passes data between them."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path

import irflow.io
from irflow.steps.field_line_tracing import FieldLineTracing
from irflow.steps.thermal_model import ThermalModel

Step = FieldLineTracing | ThermalModel


class IRFlow:
    """Runs a chain of simulation steps for one case.

    Each step runs in its own directory ``<root_dir>/<case_name>/<step.dir_name>``, and its
    result is saved there in every format of ``save_formats`` (see :mod:`irflow.io`).
    After a step finishes, its result is handed to the next step, unless that step was
    given its own input data.
    """

    def __init__(
        self,
        case_name: str,
        root_dir: str | Path,
        steps: list[Step],
        overwrite: bool = False,
        save_formats: Sequence[str] = ("vtk",),
    ) -> None:
        if not case_name or case_name in (".", "..") or "/" in case_name:
            raise ValueError(f"case_name must be a plain directory name, got {case_name!r}")

        self.case_name = case_name
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.case_dir = self.root_dir / case_name
        self.steps = steps
        self.overwrite = overwrite
        self.save_formats = irflow.io.check_formats(save_formats)

    def prepare_case_dir(self) -> None:
        """Create the case directory; replace an existing one only if ``overwrite`` is set."""
        if self.case_dir.exists():
            if not self.overwrite:
                raise FileExistsError(
                    f"Case directory already exists: {self.case_dir}. "
                    "Use IRFlow(..., overwrite=True) to replace it."
                )
            shutil.rmtree(self.case_dir)

        self.case_dir.mkdir(parents=True)
        print("Created", self.case_dir)

    def run(self) -> None:
        """Run all enabled steps in order."""
        self.prepare_case_dir()

        for index, step in enumerate(self.steps):
            if not step.enabled:
                print(f"Skipping disabled step '{step.name}'")
                continue

            step_dir = self.case_dir / step.dir_name
            step_dir.mkdir(parents=True, exist_ok=True)
            step.run(step_dir)

            result = step.get_irflow_pass_data()
            for path in irflow.io.save(result, step_dir, self.save_formats):
                print(f"Saved result of '{step.name}' to {path}")

            next_step = self.steps[index + 1] if index + 1 < len(self.steps) else None
            if next_step is None or not next_step.enabled:
                continue
            if next_step.has_own_input:
                print(f"Step '{next_step.name}' uses its own input data")
            else:
                next_step.set_irflow_pass_data(result)
