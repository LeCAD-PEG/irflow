"""Text templates for OpenFOAM case files and run scripts."""

from __future__ import annotations

import shlex
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import TextIO

TEMPLATES_PACKAGE = "irflow.backends.tm.openfoam.template_files"


def read_template(file_name: str) -> str:
    """Return the text of a template file shipped with the package."""
    return resources.files(TEMPLATES_PACKAGE).joinpath(file_name).read_text(encoding="utf-8")


def fill(text: str, values: Mapping[str, object]) -> str:
    """Replace each ``@NAME@`` in ``text`` by ``str(values[NAME])``.

    ``@NAME@`` is used instead of ``{}`` or ``$`` because OpenFOAM files use both.
    """
    for name, value in values.items():
        placeholder = f"@{name}@"
        if placeholder not in text:
            raise KeyError(f"Placeholder {placeholder} not found in template")
        text = text.replace(placeholder, str(value))
    return text


def fill_template(file_name: str, values: Mapping[str, object] | None = None) -> str:
    """Return a template file from ``template_files/`` with its placeholders filled in."""
    return fill(read_template(file_name), values or {})


def write_openfoam_plasma_bc(file_to_write: TextIO, initial_temperature: float) -> None:
    """Write the body of the coded heat-flux BC that reads ``constant/data.txt``.

    The BC interpolates the heat flux from the field line tracing points to the patch
    faces and applies it as a temperature gradient ``q / k(T)``.
    """
    file_to_write.write(
        fill_template("plasma_bc_coded_mixed.txt", {"INITIAL_TEMPERATURE": initial_temperature})
    )


def write_openfoam_sh_script(file_to_write: TextIO, of_case_path: Path, bashrc: Path) -> None:
    """Write a bash script header that enters the case and loads the OpenFOAM environment."""
    file_to_write.write(
        "#!/bin/bash\n"
        f"cd {shlex.quote(str(of_case_path))}\n"
        "set +e  # Temporarily allow non-zero exit codes while sourcing OpenFOAM bashrc\n"
        f"source {shlex.quote(str(bashrc))}\n"
        "set -e  # Exit immediately if any command fails from here on\n"
        "\n"
        'if [ -z "$WM_PROJECT_VERSION" ]; then\n'
        '    echo "ERROR: OpenFOAM environment not loaded."\n'
        "    exit 1\n"
        "fi\n"
        "\n"
        'echo "Running with OpenFOAM version $WM_PROJECT_VERSION"\n'
        "\n"
    )
