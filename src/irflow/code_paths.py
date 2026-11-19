"""Locations of external codes (OpenFOAM, ElmerFEM, ...) on this machine.

The settings live in a TOML file, separate from case configs, so that case scripts stay
portable between machines. The file is searched in this order:

1. the path given to :func:`load_code_paths`
2. the path in the environment variable ``IRFLOW_CODE_PATHS``
3. ``~/.config/irflow/code_paths.toml``

If no file is found, every code is expected to be on ``PATH``.
See ``templates/code_paths.toml`` for an example.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

CODE_PATHS_ENV_VAR = "IRFLOW_CODE_PATHS"
DEFAULT_CODE_PATHS_FILE = Path.home() / ".config" / "irflow" / "code_paths.toml"


def find_code_paths_file(path: str | Path | None = None) -> Path | None:
    """Return the code-paths file to use, or None if there is none."""
    if path is not None:
        path = Path(path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Code-paths file does not exist: {path}")
        return path

    env_path = os.environ.get(CODE_PATHS_ENV_VAR)
    if env_path:
        path = Path(env_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"Code-paths file from ${CODE_PATHS_ENV_VAR} does not exist: {path}"
            )
        return path

    if DEFAULT_CODE_PATHS_FILE.is_file():
        return DEFAULT_CODE_PATHS_FILE

    return None


def load_code_paths(path: str | Path | None = None) -> dict[str, dict[str, str]]:
    """Read the code-paths file into ``{code: {key: value}}``; empty if there is no file."""
    file_path = find_code_paths_file(path)
    if file_path is None:
        return {}

    with open(file_path, "rb") as f:
        return tomllib.load(f)


def get_code_path(code: str, key: str, path: str | Path | None = None) -> str | None:
    """Return one setting, e.g. ``get_code_path("openfoam", "bashrc")``, or None if unset."""
    value = load_code_paths(path).get(code, {}).get(key)
    if value is None:
        return None
    return str(Path(value).expanduser())
