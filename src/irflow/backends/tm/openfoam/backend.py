"""Thermal model backend for OpenFOAM (multi-region solid case run with foamMultiRun)."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from irflow.backends.tm.openfoam.case_builder import OpenFOAMCaseBuilder
from irflow.backends.tm.openfoam.templates import write_openfoam_sh_script
from irflow.code_paths import get_code_path


def resolve_openfoam_bashrc(openfoam_bashrc: str | None) -> Path:
    """Return the OpenFOAM bashrc from the step config, else from the code-paths file."""
    bashrc = openfoam_bashrc or get_code_path("openfoam", "bashrc")
    if bashrc is None:
        raise ValueError(
            "OpenFOAM bashrc is not set. Add '[openfoam] bashrc = ...' to the code-paths file "
            "(see templates/code_paths.toml) or set 'openfoam_bashrc' in the thermal model config."
        )

    bashrc_path = Path(bashrc).expanduser()
    if not bashrc_path.is_file():
        raise FileNotFoundError(f"OpenFOAM bashrc does not exist: {bashrc_path}")
    return bashrc_path


def run_script(script_path: Path) -> subprocess.CompletedProcess[str]:
    """Run a shell script, print its output and raise if it fails."""
    result = subprocess.run(
        [str(script_path)], capture_output=True, text=True, stdin=subprocess.DEVNULL
    )
    print(result.stdout)
    print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"{script_path.name} failed with return code {result.returncode}\n"
            f"--- stderr ---\n{result.stderr}\n"
            "--- stdout (last 50 lines) ---\n" + "\n".join(result.stdout.splitlines()[-50:])
        )
    return result


def list_time_dirs(of_case_path: Path) -> list[Path]:
    """Return the OpenFOAM time directories of a case, sorted by time."""
    time_dirs = []
    for path in of_case_path.iterdir():
        if not path.is_dir():
            continue
        try:
            float(path.name)
        except ValueError:
            continue
        time_dirs.append(path)
    return sorted(time_dirs, key=lambda p: float(p.name))


def read_patch_temperature(of_case_path: Path, region: str, patch: str) -> dict[str, np.ndarray]:
    """Read nodal temperature on ``patch`` for every written time from foamToVTK output.

    Returns ``{"nodes": (N, 3), "cells_2d": (M, 3), "times": (Nt,), "temperature": (Nt, N)}``.
    """
    import pyvista as pv

    # foamToVTK numbers files by time index (0, 1, 2, ...), not by simulation time
    vtk_patch_dir = of_case_path / "VTK" / region / patch
    vtk_files = sorted(
        vtk_patch_dir.glob(f"{patch}_*.vtk"), key=lambda p: int(p.stem.rsplit("_", 1)[-1])
    )
    if not vtk_files:
        raise FileNotFoundError(f"No VTK files found in {vtk_patch_dir}")

    times = np.array([float(p.name) for p in list_time_dirs(of_case_path)])
    if len(times) != len(vtk_files):
        raise RuntimeError(
            f"Found {len(times)} time directories but {len(vtk_files)} VTK files in "
            f"{vtk_patch_dir}; cannot match files to times."
        )

    surface = pv.read(vtk_files[0]).extract_surface().triangulate()
    temperature = np.empty((len(vtk_files), surface.n_points))
    for i, vtk_file in enumerate(vtk_files):
        temperature[i] = pv.read(vtk_file).point_data["T"]

    return {
        "nodes": np.asarray(surface.points, dtype=float),
        "cells_2d": np.asarray(surface.regular_faces, dtype=int),
        "times": times,
        "temperature": temperature,
    }


class BackendOpenFOAM:
    """Connector between the irflow thermal model step and OpenFOAM."""

    def __init__(
        self,
        mesh_file: str,
        mesh_type: str,
        plasma_bc_name: str,
        time_stepping: dict[str, Any],
        openfoam_bashrc: str | None = None,
        openfoam_case_name: str = "openfoam_case",
        initial_temperature: float = 300.0,
        boundary_conditions: dict[str, Any] | None = None,
        material_properties: dict[str, Any] | None = None,
    ) -> None:
        self.mesh_file = mesh_file
        self.mesh_type = mesh_type
        self.plasma_bc_name = plasma_bc_name
        self.time_stepping = time_stepping
        self.openfoam_bashrc = resolve_openfoam_bashrc(openfoam_bashrc)
        self.openfoam_case_name = openfoam_case_name
        self.initial_temperature = initial_temperature
        self.boundary_conditions = boundary_conditions
        self.material_properties = material_properties

        self.is_transient = time_stepping["type"].lower() != "steady"
        self.time_step = time_stepping["time_step"] if self.is_transient else None
        self.end_time = time_stepping["end_time"] if self.is_transient else None

        self.irflow_pass_data: dict[str, np.ndarray] | None = None
        self.temperature: dict[str, np.ndarray] | None = None

    def set_irflow_pass_data(self, irflow_pass_data: dict[str, np.ndarray]) -> None:
        """Heat flux from the previous step: ``{"xyz": (M, 3), "heat_flux": (M,)}``."""
        self.irflow_pass_data = irflow_pass_data

    def run(self, step_dir: Path) -> None:
        """Build the OpenFOAM case in ``step_dir``, run it and read the plasma-patch temperature."""
        step_dir = Path(step_dir)
        if not step_dir.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {step_dir}")
        if self.irflow_pass_data is None:
            raise RuntimeError(
                "OpenFOAM backend has no heat flux input: run a field line tracing step "
                "before the thermal model."
            )

        builder = OpenFOAMCaseBuilder(
            case_root=step_dir,
            case_name=self.openfoam_case_name,
            time_settings=self.time_stepping,
            mesh_type=self.mesh_type,
            plasma_bc_name=self.plasma_bc_name,
            initial_temperature=self.initial_temperature,
            boundary_conditions=self.boundary_conditions,
            material_properties=self.material_properties,
        )
        builder.build_case(self.irflow_pass_data["xyz"], self.irflow_pass_data["heat_flux"])
        of_case_path = builder.case_path

        # 1) Convert mesh and split it into regions by cell zones
        script_prepare_path = of_case_path / "prepare.sh"
        with open(script_prepare_path, "w") as script:
            write_openfoam_sh_script(script, of_case_path, self.openfoam_bashrc)
            script.write(
                f"ideasUnvToFoam {shlex.quote(self.mesh_file)} "
                f"-case {shlex.quote(str(of_case_path))}\n"
            )
            script.write("splitMeshRegions -cellZones -overwrite\n")
        script_prepare_path.chmod(0o755)
        run_script(script_prepare_path)

        # 2) Write fields and dictionaries that depend on the regions
        builder.write_T()
        builder.write_controlDict(regions=True)
        builder.write_physicalProperties(regions=True)
        builder.write_fvSchemes(regions=True)
        builder.write_fvSolution(regions=True)

        # 3) Solve and export the plasma-facing patch to VTK
        region_map = builder.get_region_boundary_map()
        plasma_region = next(
            (region for region, patches in region_map.items() if self.plasma_bc_name in patches),
            None,
        )
        if plasma_region is None:
            raise ValueError(
                f"Plasma boundary '{self.plasma_bc_name}' not found in any region: {region_map}"
            )
        other_patches = [p for p in region_map[plasma_region] if p != self.plasma_bc_name]

        script_run_path = of_case_path / "run.sh"
        with open(script_run_path, "w") as script:
            write_openfoam_sh_script(script, of_case_path, self.openfoam_bashrc)
            script.write("foamMultiRun\n")
            script.write(
                f"foamToVTK -region {plasma_region} "
                f"-excludePatches '({' '.join(other_patches)})'\n"
            )
        script_run_path.chmod(0o755)

        # Empty '.foam' file lets ParaView/pyvista open the case
        (of_case_path / "case.foam").touch()

        run_script(script_run_path)
        print("foamMultiRun finished!")

        temperature = read_patch_temperature(of_case_path, plasma_region, self.plasma_bc_name)
        if not self.is_transient:
            # steady state: only the converged solution is compared with the IR image
            temperature["times"] = temperature["times"][-1:]
            temperature["temperature"] = temperature["temperature"][-1:]
        self.temperature = temperature

    def get_temperature(self) -> dict[str, np.ndarray]:
        """Plasma-facing surface temperature, see :func:`read_patch_temperature`."""
        if self.temperature is None:
            raise RuntimeError("OpenFOAM has not been run yet.")
        return self.temperature
