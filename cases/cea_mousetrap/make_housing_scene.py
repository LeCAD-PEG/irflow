"""Build the housing surfaces for the IR camera scene of ``gyrohit_openfoam_raysect.py``.

Converts the thermal model mesh with OpenFOAM (ideasUnvToFoam + splitMeshRegions, in a
temporary directory) and keeps the outer surfaces of the housing regions, i.e. all their
patches except the interfaces to other regions. The result is written to
``cases/cea_mousetrap/data/housing_surface.vtp``. Run it once; it takes a few seconds.
"""

import shlex
import tempfile
from pathlib import Path

import numpy as np

from irflow.backends.tm.openfoam.backend import resolve_openfoam_bashrc, run_script, write_script
from irflow.backends.tm.openfoam.case_builder import OpenFOAMCaseBuilder
from irflow.backends.tm.openfoam.polymesh import merge_surfaces, read_boundary_surface
from irflow.io.vtk import to_polydata

MESH_FILE = "/home/brankm/work/CEA_mousetrap/Mesh_groups_translated.unv"
HOUSING_REGIONS = ["housing_1", "housing_2"]
OUTPUT_FILE = Path(__file__).resolve().parent / "data" / "housing_surface.vtp"
WORK_DIR = "/home/brankm/data/irflow_studies"  # temporary files on the data disk


def main() -> None:
    bashrc = resolve_openfoam_bashrc(None)  # from the code-paths file

    with tempfile.TemporaryDirectory(dir=WORK_DIR, prefix="housing_scene_") as tmp:
        builder = OpenFOAMCaseBuilder(
            case_root=tmp,
            case_name="mesh",
            time_settings={"type": "steady"},
            mesh_type="unv",
            plasma_bc_name="plasmaBC",
        )
        builder.build_case(np.zeros((0, 3)), np.zeros(0))  # just enough for the utilities
        case = builder.case_path
        script = write_script(
            case / "split_mesh.sh",
            case,
            bashrc,
            [
                f"ideasUnvToFoam {shlex.quote(MESH_FILE)} -case {shlex.quote(str(case))}",
                "splitMeshRegions -cellZones -overwrite",
            ],
        )
        run_script(script)

        surfaces = [
            read_boundary_surface(case / "constant" / region / "polyMesh")
            for region in HOUSING_REGIONS
        ]

    housing = merge_surfaces(surfaces)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    to_polydata(housing).save(OUTPUT_FILE)
    print(f"Wrote {housing.n_cells} triangles of {HOUSING_REGIONS} to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
