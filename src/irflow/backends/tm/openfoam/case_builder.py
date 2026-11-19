"""Writes the files of an OpenFOAM multi-region solid case (foamMultiRun).

Whole dictionaries live as text templates in ``template_files/`` (``@NAME@`` placeholders);
this module fills them in. ``tests/test_openfoam_golden.py`` checks the written files byte
for byte.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from irflow.backends.tm.openfoam.templates import fill, fill_template, read_template

# OpenFOAM model names written for each material property type.
# NOTE: the table/polynomial names are kept as they were and have not been checked against
# OpenFOAM yet; only constant properties have been run (see CLAUDE.md, known issues).
DENSITY_MODELS = {"constant": "rhoConst", "table": "eIcoTabulated", "polynomial": "ePolynomial"}
HEAT_CAPACITY_MODELS = {"constant": "eConst", "table": "eIcoTabulated", "polynomial": "hPolynomial"}
HEAT_CONDUCTIVITY_MODELS = {
    "constant": "constIsoSolid",
    "table": "tabulatedSolidTransport",
    "polynomial": "hPolynomial",
    "polynomialSolid": "hPolynomial",
}

# Used when no material properties are given (SiO2-like)
DEFAULT_MODEL_LINES = (
    "    transport       polynomialSolid;\n"
    "    thermo          eConst;\n"
    "    equationOfState rhoConst;\n"
)
DEFAULT_DENSITY = "        rho             3060;// kg/m3\n"
DEFAULT_HEAT_CAPACITY = "        Cv              900;\n"
DEFAULT_HEAT_CONDUCTIVITY = (
    "        kappaCoeffs<8>  (  1.82e+01  -7.40e-03   1.20e-05  -8.50e-09   0 0 0 0 );\n"
)

# Entries of the boundaryField in 0/<region>/T, one per kind of patch
COUPLED_PATCH = (
    "    @NAME@\n"
    "    {\n"
    "        type            coupledTemperature;\n"
    "        value           uniform @INITIAL_TEMPERATURE@;\n"
    "    }\n"
    "\n"
)
PLASMA_PATCH = "       @NAME@\n{\n"  # followed by the coded BC from plasma_bc_coded_mixed.txt
HTC_PATCH = (
    "    @NAME@\n"
    "    {\n"
    "        type            externalWallHeatFluxTemperature;\n"
    "        mode            flux;\n"
    "        kappaMethod     solidThermo;\n"
    "        h               uniform @VALUE@;\n"
    "        Ta              uniform @T_EXT@;\n"
    "        value           uniform @INITIAL_TEMPERATURE@;\n"
    "   }\n"
    "\n"
)
HEAT_FLUX_PATCH = (
    "    @NAME@\n"
    "    {\n"
    "        type            externalWallHeatFluxTemperature;\n"
    "        mode            flux;\n"
    "        kappaMethod     solidThermo;\n"
    "        q               uniform @VALUE@;\n"
    "        value           uniform @INITIAL_TEMPERATURE@;\n"
    "   }\n"
    "\n"
)
ZERO_GRADIENT_PATCH = "    @NAME@\n    {\n    type            zeroGradient;\n   }\n\n"
TEMPERATURE_PATCH = (
    "    @NAME@\n"
    "    {\n"
    "    type            fixedValue;\n"
    "    value           uniform @VALUE@;\n"
    "   }\n"
    "\n"
)

DEFAULT_FACES_CONDITION = {"type": "heat_flux", "value_type": "constant", "value": "zeroGradient"}


def property_entry(keyword: str, spec: dict[str, Any]) -> str:
    """Text of one material property (``rho``, ``Cv`` or ``kappa``) in physicalProperties."""
    kind = spec.get("type")
    value: Any = spec.get("value")
    if kind == "constant":
        return f"        {keyword}             {value};\n"
    if kind == "table":
        rows = "".join(f"            ({row[0]} {row[1]})\n" for row in value)
        return f"         {keyword}\n{{    values\n    (\n{rows}    )\n}}\n"
    if kind in ("polynomial", "polynomialSolid"):
        coefficients = "".join(f" {coefficient}" for coefficient in value)
        return f"{keyword}Coeffs<8>     ({coefficients});"
    raise ValueError(f"Unknown type {kind!r} for material property '{keyword}'")


def model_name(models: dict[str, str], spec: dict[str, Any], name: str) -> str:
    """OpenFOAM model name for a material property spec ``{"type": ..., "value": ...}``."""
    try:
        return models[spec["type"]]
    except KeyError:
        raise ValueError(
            f"Unknown type {spec.get('type')!r} for {name}; supported: {sorted(models)}"
        ) from None


class OpenFOAMCaseBuilder:
    """Writes dictionaries and fields of a multi-region solid case."""

    def __init__(
        self,
        case_root: str | Path,
        case_name: str,
        time_settings: dict[str, Any],
        mesh_type: str,
        plasma_bc_name: str,
        initial_temperature: float = 300.0,
        boundary_conditions: dict[str, Any] | None = None,
        material_properties: dict[str, Any] | None = None,
    ) -> None:
        self.case_root = Path(case_root)
        self.case_name = case_name
        self.time_settings = time_settings
        self.mesh_type = mesh_type
        self.plasma_bc_name = plasma_bc_name
        self.initial_temperature = initial_temperature
        self.boundary_conditions = boundary_conditions or {}
        self.material_properties = material_properties or {}

        self.case_path = self.case_root / self.case_name
        self.constant_path = self.case_path / "constant"
        self.system_path = self.case_path / "system"
        self.zero_path = self.case_path / "0"

    def get_region_boundary_map(self) -> dict[str, list[str]]:
        """Return ``{region: [patch names]}`` read from ``constant/<region>/polyMesh/boundary``."""
        region_map = {}
        exclude = {"polyMesh", "regionProperties"}

        for region_dir in sorted(self.constant_path.iterdir()):
            if not region_dir.is_dir() or region_dir.name in exclude:
                continue

            boundary_file = region_dir / "polyMesh" / "boundary"
            if not boundary_file.exists():
                continue

            boundaries = []
            previous = ""
            with open(boundary_file) as f:
                for line in f:
                    stripped = line.strip()
                    # A patch name is the last non-empty line before "{"
                    if stripped == "{" and previous and not previous.startswith("//"):
                        boundaries.append(previous)
                    if stripped:
                        previous = stripped

            if "FoamFile" in boundaries:
                boundaries.remove("FoamFile")
            region_map[region_dir.name] = boundaries

        return region_map

    def create_directory_structure(self) -> None:
        for path in (self.case_path, self.constant_path, self.system_path, self.zero_path):
            path.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Dictionaries
    # ---------------------------------------------------------
    def write_controlDict(self, regions: bool = False) -> None:
        region_solvers = ""
        if regions:
            lines = "".join(f"   {region}      solid;\n" for region in self.get_region_boundary_map())
            region_solvers = f"regionSolvers\n{{\n{lines}}}\n"

        if self.time_settings["type"] == "steady":
            end_time, delta_t = 1, 1
        else:
            end_time, delta_t = self.time_settings["end_time"], self.time_settings["time_step"]

        text = fill_template(
            "controlDict.txt",
            {"REGION_SOLVERS": region_solvers, "END_TIME": end_time, "DELTA_T": delta_t},
        )
        (self.system_path / "controlDict").write_text(text)

    def write_fvSchemes(self, regions: bool = False) -> None:
        """Write ``system/fvSchemes``, or ``system/<region>/fvSchemes`` for every region."""
        text = read_template("fvSchemes.txt")
        for directory in self.system_directories(regions):
            (directory / "fvSchemes").write_text(text)

    def write_fvSolution(self, regions: bool = False) -> None:
        """Write ``system/fvSolution``, or ``system/<region>/fvSolution`` for every region."""
        if not regions:
            values = {
                "LOCATION": "system/heater",
                "TOLERANCE": "0.01",
                "REL_TOL": "0.05",
                "PIMPLE_SETTINGS": "\tnNonOrthogonalCorrectors 1;  "
                "// mesh has faces up to 89.7 degrees non-orthogonality\n",
            }
            (self.system_path / "fvSolution").write_text(fill_template("fvSolution.txt", values))
            return

        for directory in self.system_directories(regions):
            values = {
                "LOCATION": f"system/{directory.name}",
                "TOLERANCE": "1e-6",
                "REL_TOL": "0.01",
                "PIMPLE_SETTINGS": "\tnNonOrthogonalCorrectors 3;  "
                "// mesh has faces up to 89.7 degrees non-orthogonality\n"
                "   nOuterCorrectors            50;\n",
            }
            (directory / "fvSolution").write_text(fill_template("fvSolution.txt", values))

    def system_directories(self, regions: bool) -> list[Path]:
        """``system/`` itself, or ``system/<region>/`` for every region (created if needed)."""
        if not regions:
            return [self.system_path]

        directories = [self.system_path / region for region in self.get_region_boundary_map()]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        return directories

    def write_regionProperties(self) -> None:
        (self.constant_path / "regionProperties").write_text(read_template("regionProperties.txt"))

    def write_physicalProperties(self, regions: bool = False) -> None:
        """Write ``constant/<region>/physicalProperties`` for every region (multi-region only)."""
        if not regions:
            return

        for region in self.get_region_boundary_map():
            if self.material_properties:
                if region not in self.material_properties:
                    raise KeyError(f"No material properties given for region '{region}'")
                material = self.material_properties[region]
                density = material["density"]
                heat_capacity = material["heat_capacity"]
                heat_conductivity = material["heat_conductivity"]

                transport = model_name(HEAT_CONDUCTIVITY_MODELS, heat_conductivity, "conductivity")
                thermo = model_name(HEAT_CAPACITY_MODELS, heat_capacity, "heat capacity")
                state = model_name(DENSITY_MODELS, density, "density")
                values = {
                    "MODEL_LINES": f"    transport       {transport};\n"
                    f"    thermo       {thermo};\n"
                    f"    equationOfState       {state};\n",
                    "DENSITY": property_entry("rho", density),
                    "HEAT_CAPACITY": property_entry("Cv", heat_capacity),
                    "HEAT_CONDUCTIVITY": property_entry("kappa", heat_conductivity),
                }
            else:
                values = {
                    "MODEL_LINES": DEFAULT_MODEL_LINES,
                    "DENSITY": DEFAULT_DENSITY,
                    "HEAT_CAPACITY": DEFAULT_HEAT_CAPACITY,
                    "HEAT_CONDUCTIVITY": DEFAULT_HEAT_CONDUCTIVITY,
                }

            values["REGION"] = region
            text = fill_template("physicalProperties.txt", values)
            (self.constant_path / region / "physicalProperties").write_text(text)

    # ---------------------------------------------------------
    # Heat flux / boundary data
    # ---------------------------------------------------------
    def write_heat_flux(self, xyz: np.ndarray, heat_flux: np.ndarray) -> None:
        """Write ``constant/data.txt`` (x y z q per line) read by the coded plasma BC."""
        data = np.column_stack([xyz, heat_flux])
        np.savetxt(self.constant_path / "data.txt", data, header="time: 0.0", comments="")

    def build_case(self, xyz: np.ndarray, heat_flux: np.ndarray) -> None:
        """Write the region-independent part of the case (before the mesh is split)."""
        self.create_directory_structure()
        self.write_controlDict()
        self.write_fvSchemes()
        self.write_fvSolution()
        self.write_regionProperties()
        self.write_heat_flux(xyz, heat_flux)

    # ---------------------------------------------------------
    # Fields
    # ---------------------------------------------------------
    def patch_entry(self, region: str, patch: str) -> str:
        """The boundaryField entry of ``patch`` in ``0/<region>/T``."""
        values: dict[str, object] = {"NAME": patch, "INITIAL_TEMPERATURE": self.initial_temperature}

        if region + "_to_" in patch:
            return fill(COUPLED_PATCH, values)
        if patch == self.plasma_bc_name:
            return fill(PLASMA_PATCH, {"NAME": patch}) + fill_template(
                "plasma_bc_coded_mixed.txt", {"INITIAL_TEMPERATURE": self.initial_temperature}
            )

        condition = self.boundary_conditions.get(patch)
        if condition is None and patch == "defaultFaces":
            condition = DEFAULT_FACES_CONDITION
        if condition is None:
            raise KeyError(f"No boundary condition given for patch '{patch}' (region '{region}')")

        kind, value = condition["type"], condition["value"]
        if kind in ("htc", "heat_flux") and value != "zeroGradient":
            if condition.get("value_type") != "constant":
                raise NotImplementedError(
                    f"Patch '{patch}': only value_type 'constant' is supported for {kind}"
                )
            if kind == "htc":
                return fill(HTC_PATCH, {**values, "VALUE": value, "T_EXT": condition["T_ext"]})
            return fill(HEAT_FLUX_PATCH, {**values, "VALUE": value})
        if value == "zeroGradient":
            return fill(ZERO_GRADIENT_PATCH, {"NAME": patch})
        if kind == "temperature":
            return fill(TEMPERATURE_PATCH, {"NAME": patch, "VALUE": value})
        raise ValueError(f"Patch '{patch}': unknown boundary condition type {kind!r}")

    def write_T(self) -> None:
        """Write the initial temperature field ``0/<region>/T`` for every region."""
        for region, patches in self.get_region_boundary_map().items():
            boundary_fields = "".join(self.patch_entry(region, patch) for patch in patches)
            text = fill_template(
                "T.txt",
                {
                    "INITIAL_TEMPERATURE": self.initial_temperature,
                    "BOUNDARY_FIELDS": boundary_fields,
                },
            )
            (self.zero_path / region / "T").write_text(text, encoding="utf-8")
