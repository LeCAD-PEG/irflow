"""Thermal model backend for OpenFOAM (multi-region solid case run with foamMultiRun)."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from irflow.backends.base import ThermalModelBackend
from irflow.backends.tm.openfoam.case_builder import OpenFOAMCaseBuilder
from irflow.backends.tm.openfoam.templates import write_openfoam_sh_script
from irflow.code_paths import get_code_path
from irflow.data import HeatFluxField, SurfaceMesh, TemperatureField
from irflow.mapping import check_overlap


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


BOUNDARY_CONDITION_TYPES = ("heat_flux", "htc", "temperature")


def check_boundary_conditions(boundary_conditions: dict[str, Any]) -> None:
    """Raise ValueError for boundary conditions the case builder cannot write."""
    for patch, condition in boundary_conditions.items():
        if condition.get("type") not in BOUNDARY_CONDITION_TYPES:
            raise ValueError(
                f"Boundary condition '{patch}': type must be one of "
                f"{BOUNDARY_CONDITION_TYPES}, got {condition.get('type')!r}"
            )


def write_script(path: Path, of_case_path: Path, bashrc: Path, commands: list[str]) -> Path:
    """Write an executable bash script that loads OpenFOAM and runs ``commands`` in the case."""
    with open(path, "w") as script:
        write_openfoam_sh_script(script, of_case_path, bashrc)
        for command in commands:
            script.write(command + "\n")
    path.chmod(0o755)
    return path


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


def list_patch_vtk_files(of_case_path: Path, region: str, patch: str) -> list[Path]:
    """foamToVTK files of one patch, sorted by time index (files are numbered 0, 1, 2, ...)."""
    vtk_patch_dir = of_case_path / "VTK" / region / patch
    vtk_files = sorted(
        vtk_patch_dir.glob(f"{patch}_*.vtk"), key=lambda p: int(p.stem.rsplit("_", 1)[-1])
    )
    if not vtk_files:
        raise FileNotFoundError(f"No VTK files found in {vtk_patch_dir}")
    return vtk_files


def read_patch_surface(vtk_file: Path) -> tuple[SurfaceMesh, pv.PolyData]:
    """Read a foamToVTK patch file as a triangular surface."""
    surface = pv.read(vtk_file).extract_surface().triangulate()
    return SurfaceMesh(nodes=np.asarray(surface.points), triangles=surface.regular_faces), surface


def read_patch_temperature(of_case_path: Path, region: str, patch: str) -> TemperatureField:
    """Read nodal temperature on ``patch`` for every written time from foamToVTK output."""
    vtk_files = list_patch_vtk_files(of_case_path, region, patch)
    times = np.array([float(p.name) for p in list_time_dirs(of_case_path)])
    if len(times) != len(vtk_files):
        raise RuntimeError(
            f"Found {len(times)} time directories but {len(vtk_files)} VTK files for patch "
            f"'{patch}'; cannot match files to times."
        )

    mesh, _ = read_patch_surface(vtk_files[0])
    temperature = np.empty((len(vtk_files), mesh.n_nodes))
    for i, vtk_file in enumerate(vtk_files):
        _, surface = read_patch_surface(vtk_file)
        temperature[i] = surface.point_data["T"]

    return TemperatureField(mesh=mesh, times=times, temperature=temperature)


class BackendOpenFOAM(ThermalModelBackend):
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
        check_boundary_conditions(boundary_conditions or {})
        self.material_properties = material_properties

        self.is_transient = time_stepping["type"].lower() != "steady"
        self.time_step = time_stepping["time_step"] if self.is_transient else None
        self.end_time = time_stepping["end_time"] if self.is_transient else None

        self.result: TemperatureField | None = None

    def run(
        self, step_dir: Path, heat_flux: HeatFluxField, mapping_tolerance: float | None = None
    ) -> None:
        """Build the OpenFOAM case in ``step_dir``, run it and read the plasma-patch temperature."""
        step_dir = Path(step_dir)
        if not step_dir.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {step_dir}")

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
        builder.build_case(heat_flux.mesh.cell_centers(), heat_flux.heat_flux)
        of_case_path = builder.case_path
        bashrc = self.openfoam_bashrc

        # 1) Convert mesh and split it into regions by cell zones
        run_script(
            write_script(
                of_case_path / "prepare.sh",
                of_case_path,
                bashrc,
                [
                    f"ideasUnvToFoam {shlex.quote(self.mesh_file)} "
                    f"-case {shlex.quote(str(of_case_path))}",
                    "splitMeshRegions -cellZones -overwrite",
                ],
            )
        )

        # 2) Write fields and dictionaries that depend on the regions
        builder.write_T()
        builder.write_controlDict(regions=True)
        builder.write_physicalProperties(regions=True)
        builder.write_fvSchemes(regions=True)
        builder.write_fvSolution(regions=True)

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
        export_plasma_patch = (
            f"foamToVTK -region {plasma_region} -excludePatches '({' '.join(other_patches)})'"
        )

        # 3) Check that the plasma patch lies on the heat flux mesh before solving
        run_script(
            write_script(
                of_case_path / "export_plasma_patch.sh",
                of_case_path,
                bashrc,
                [export_plasma_patch + " -time 0 -noInternal"],
            )
        )
        initial_files = list_patch_vtk_files(of_case_path, plasma_region, self.plasma_bc_name)
        plasma_mesh, _ = read_patch_surface(initial_files[0])
        distance = check_overlap(heat_flux.mesh, plasma_mesh, mapping_tolerance)
        print(f"Plasma patch lies within {distance:.3g} m of the heat flux surface.")

        # 4) Solve and export the plasma-facing patch to VTK for every time
        (of_case_path / "case.foam").touch()  # lets ParaView/pyvista open the case
        run_script(
            write_script(
                of_case_path / "run.sh",
                of_case_path,
                bashrc,
                ["foamMultiRun", export_plasma_patch],
            )
        )
        print("foamMultiRun finished!")

        result = read_patch_temperature(of_case_path, plasma_region, self.plasma_bc_name)
        if not self.is_transient:
            # steady state: only the converged solution is compared with the IR image
            result = TemperatureField(result.mesh, result.times[-1:], result.temperature[-1:])
        self.result = result

    def get_result(self) -> TemperatureField:
        """Plasma-facing surface temperature for every written time."""
        if self.result is None:
            raise RuntimeError("OpenFOAM has not been run yet.")
        return self.result
