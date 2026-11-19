"""Top-level workflow: runs the simulation steps in order and passes data between them."""

from __future__ import annotations

import shutil
from pathlib import Path

from irflow.steps.field_line_tracing import FieldLineTracing
from irflow.steps.thermal_model import ThermalModel

Step = FieldLineTracing | ThermalModel


class IRFlow:
    """Runs a chain of simulation steps for one case.

    Each step runs in its own directory ``<root_dir>/<case_name>/<step.dir_name>``.
    After a step finishes, its output is handed to the next step.
    """

    def __init__(
        self,
        case_name: str,
        root_dir: str | Path,
        steps: list[Step],
        overwrite: bool = False,
    ) -> None:
        if not case_name or case_name in (".", "..") or "/" in case_name:
            raise ValueError(f"case_name must be a plain directory name, got {case_name!r}")

        self.case_name = case_name
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.case_dir = self.root_dir / case_name
        self.steps = steps
        self.overwrite = overwrite

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

            next_step = self.steps[index + 1] if index + 1 < len(self.steps) else None
            if next_step is not None and next_step.enabled:
                next_step.set_irflow_pass_data(step.get_irflow_pass_data())
