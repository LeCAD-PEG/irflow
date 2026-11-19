"""Text templates for OpenFOAM case files and run scripts."""

from __future__ import annotations

import shlex
from importlib import resources
from pathlib import Path
from typing import TextIO

TEMPLATES_PACKAGE = "irflow.backends.tm.openfoam.template_files"


def read_template(file_name: str) -> str:
    """Return the text of a template file shipped with the package."""
    return resources.files(TEMPLATES_PACKAGE).joinpath(file_name).read_text(encoding="utf-8")


def write_openfoam_plasma_bc(file_to_write: TextIO, initial_temperature: float) -> None:
    """Write the body of the coded heat-flux BC that reads ``constant/data.txt``.

    The BC interpolates the heat flux from the field line tracing points to the patch
    faces and applies it as a temperature gradient ``q / k(T)``.
    """
    text = read_template("plasma_bc_coded_mixed.txt")
    file_to_write.write(text.replace("@INITIAL_TEMPERATURE@", str(initial_temperature)))


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
