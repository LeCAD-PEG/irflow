"""Tests for reading the machine-specific code-paths file."""

from __future__ import annotations

from pathlib import Path

import pytest

from irflow import code_paths

CONFIG_TEXT = """
[openfoam]
bashrc = "~/OpenFOAM-dev/etc/bashrc"
"""


def write_config(path: Path) -> Path:
    path.write_text(CONFIG_TEXT)
    return path


def test_no_file_gives_empty_config() -> None:
    assert code_paths.load_code_paths() == {}
    assert code_paths.get_code_path("openfoam", "bashrc") is None


def test_explicit_path(tmp_path: Path) -> None:
    config = write_config(tmp_path / "codes.toml")
    assert code_paths.load_code_paths(config) == {
        "openfoam": {"bashrc": "~/OpenFOAM-dev/etc/bashrc"}
    }


def test_missing_explicit_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        code_paths.load_code_paths(tmp_path / "missing.toml")


def test_environment_variable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = write_config(tmp_path / "codes.toml")
    monkeypatch.setenv(code_paths.CODE_PATHS_ENV_VAR, str(config))
    assert code_paths.find_code_paths_file() == config


def test_default_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = write_config(tmp_path / "default.toml")
    monkeypatch.setattr(code_paths, "DEFAULT_CODE_PATHS_FILE", config)
    assert code_paths.find_code_paths_file() == config


def test_get_code_path_expands_home(tmp_path: Path) -> None:
    config = write_config(tmp_path / "codes.toml")
    value = code_paths.get_code_path("openfoam", "bashrc", config)
    assert value == str(Path.home() / "OpenFOAM-dev" / "etc" / "bashrc")
    assert code_paths.get_code_path("elmerfem", "bin_dir", config) is None
