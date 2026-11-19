"""Tests for IRFlow: case directory handling and passing data between steps."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from irflow import IRFlow


class RecordingStep:
    """Minimal stand-in for a simulation step that records what IRFlow does with it."""

    def __init__(self, name: str, enabled: bool = True) -> None:
        self.name = name
        self.enabled = enabled
        self.dir_name = name
        self.run_dir: Path | None = None
        self.received: dict[str, Any] | None = None

    def run(self, step_dir: Path) -> None:
        self.run_dir = step_dir

    def get_irflow_pass_data(self) -> dict[str, Any]:
        return {"from": self.name}

    def set_irflow_pass_data(self, data: dict[str, Any]) -> None:
        self.received = data


@pytest.mark.parametrize("case_name", ["", ".", "..", "a/b"])
def test_rejects_case_names_that_are_not_plain_directories(
    tmp_path: Path, case_name: str
) -> None:
    with pytest.raises(ValueError):
        IRFlow(case_name=case_name, root_dir=tmp_path, steps=[])


def test_existing_case_dir_is_kept_without_overwrite(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "result.txt").write_text("keep me")

    with pytest.raises(FileExistsError):
        IRFlow(case_name="case", root_dir=tmp_path, steps=[]).run()
    assert (case_dir / "result.txt").exists()


def test_existing_case_dir_is_replaced_with_overwrite(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "old.txt").write_text("old")

    IRFlow(case_name="case", root_dir=tmp_path, steps=[], overwrite=True).run()
    assert case_dir.is_dir()
    assert not (case_dir / "old.txt").exists()


def test_steps_run_in_own_dirs_and_pass_data_forward(tmp_path: Path) -> None:
    first, second = RecordingStep("first"), RecordingStep("second")

    IRFlow(case_name="case", root_dir=tmp_path, steps=[first, second]).run()

    assert first.run_dir == tmp_path / "case" / "first"
    assert second.run_dir == tmp_path / "case" / "second"
    assert first.run_dir.is_dir() and second.run_dir.is_dir()
    assert second.received == {"from": "first"}


def test_disabled_step_is_skipped_and_gets_no_data(tmp_path: Path) -> None:
    first, second = RecordingStep("first"), RecordingStep("second", enabled=False)

    IRFlow(case_name="case", root_dir=tmp_path, steps=[first, second]).run()

    assert second.run_dir is None
    assert second.received is None


def test_importing_irflow_does_not_import_backends() -> None:
    # Fresh interpreter, because other tests import backends into this one
    code = (
        "import sys, irflow; "
        "print(any(name.startswith('irflow.backends') for name in sys.modules))"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    assert result.stdout.strip() == "False"
