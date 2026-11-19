"""Golden-file tests: the OpenFOAM case builder must keep writing exactly the same files.

Run with ``IRFLOW_UPDATE_GOLDEN=1`` to rewrite the golden files after an intended change, then
review the diff of ``tests/data/openfoam_golden`` before committing.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from irflow.backends.tm.openfoam.case_builder import OpenFOAMCaseBuilder

GOLDEN_DIR = Path(__file__).parent / "data" / "openfoam_golden"
UPDATE = os.environ.get("IRFLOW_UPDATE_GOLDEN") == "1"

REGION_PATCHES = {
    "sample": ["plasmaBC", "const_T", "sample_to_housing", "defaultFaces"],
    "housing": ["housing_to_sample", "housing_to_screws", "cooled", "loaded"],
    "screws": ["screws_to_housing", "adiabatic"],
}

BOUNDARY_CONDITIONS = {
    "plasmaBC": {"type": "heat_flux", "value": None},
    "const_T": {"type": "temperature", "value": 373, "value_type": "constant"},
    "cooled": {"type": "htc", "value": 17283, "T_ext": 300.0, "value_type": "constant"},
    "loaded": {"type": "heat_flux", "value": 1000.0, "value_type": "constant"},
    "adiabatic": {"type": "heat_flux", "value": "zeroGradient", "value_type": "constant"},
}

TABLE = np.array([[300.0, 600.0], [1000.0, 1200.0]])
POLYNOMIAL = np.array([21.8, -0.0142, 8.3e-06, 0, 0, 0, 0, 0])

MATERIALS = {
    "sample": {
        "density": {"type": "constant", "value": 3060},
        "heat_capacity": {"type": "constant", "value": 1000},
        "heat_conductivity": {"type": "constant", "value": 5},
    },
    "housing": {
        "density": {"type": "table", "value": TABLE},
        "heat_capacity": {"type": "table", "value": TABLE},
        "heat_conductivity": {"type": "polynomial", "value": POLYNOMIAL},
    },
    "screws": {
        "density": {"type": "polynomial", "value": POLYNOMIAL},
        "heat_capacity": {"type": "polynomial", "value": POLYNOMIAL},
        "heat_conductivity": {"type": "table", "value": TABLE},
    },
}


def boundary_file(patches: list[str]) -> str:
    entries = "".join(f"    {name}\n    {{\n        type wall;\n    }}\n" for name in patches)
    return f"FoamFile\n{{\n    object boundary;\n}}\n\n{len(patches)}\n(\n{entries})\n"


def build_case(
    case_root: Path, transient: bool, materials: dict | None, split: bool = True
) -> Path:
    """Write every file the builder produces for a fake 3-region case; return the case path."""
    time_settings = (
        {"type": "transient", "time_step": 0.001, "end_time": 0.2}
        if transient
        else {"type": "steady"}
    )
    builder = OpenFOAMCaseBuilder(
        case_root=case_root,
        case_name="case",
        time_settings=time_settings,
        mesh_type="unv",
        plasma_bc_name="plasmaBC",
        initial_temperature=373.0,
        boundary_conditions=BOUNDARY_CONDITIONS,
        material_properties=materials,
    )
    xyz = np.array([[0.0, 0.0, 0.0], [0.001, 0.002, 0.003]])
    builder.build_case(xyz, np.array([1.0e6, 2.5e6]))
    if not split:
        return builder.case_path

    # what ideasUnvToFoam + splitMeshRegions would create
    for region, patches in REGION_PATCHES.items():
        poly_mesh = builder.constant_path / region / "polyMesh"
        poly_mesh.mkdir(parents=True)
        (poly_mesh / "boundary").write_text(boundary_file(patches))
        (builder.zero_path / region).mkdir(parents=True)

    builder.write_T()
    builder.write_controlDict(regions=True)
    builder.write_physicalProperties(regions=True)
    builder.write_fvSchemes(regions=True)
    builder.write_fvSolution(regions=True)
    for poly_mesh in builder.constant_path.glob("*/polyMesh"):
        (poly_mesh / "boundary").unlink()
        poly_mesh.rmdir()
    return builder.case_path


def written_files(case_path: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(case_path)): path.read_bytes()
        for path in sorted(case_path.rglob("*"))
        if path.is_file()
    }


@pytest.mark.parametrize(
    "scenario, transient, materials, split",
    [
        ("before_split", True, MATERIALS, False),
        ("transient_materials", True, MATERIALS, True),
        ("steady_default_materials", False, None, True),
    ],
)
def test_case_files_match_golden(
    tmp_path: Path, scenario: str, transient: bool, materials: dict | None, split: bool
) -> None:
    files = written_files(build_case(tmp_path, transient, materials, split))
    golden_path = GOLDEN_DIR / scenario

    if UPDATE:
        for name, content in files.items():
            (golden_path / name).parent.mkdir(parents=True, exist_ok=True)
            (golden_path / name).write_bytes(content)

    golden = written_files(golden_path)
    assert sorted(files) == sorted(golden)
    for name in files:
        assert files[name] == golden[name], f"{name} differs from the golden file"
