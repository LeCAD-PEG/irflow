"""Writes the files of an OpenFOAM multi-region solid case (foamMultiRun)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from irflow.backends.tm.openfoam.templates import write_openfoam_plasma_bc

class OpenFOAMCaseBuilder:
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
            prev = ""

            with open(boundary_file, "r") as f:
                for line in f:
                    stripped = line.strip()

                    # Patch name is always the line before "{"
                    if stripped == "{":
                        if prev and not prev.startswith("//"):
                            boundaries.append(prev)

                    # Track previous non-empty line
                    if stripped:
                        prev = stripped

            if "FoamFile" in boundaries:
                boundaries.remove("FoamFile")

            region_map[region_dir.name] = boundaries

        return region_map


    # ---------------------------------------------------------
    # Directory creation
    # ---------------------------------------------------------
    def create_directory_structure(self) -> None:
        self.case_path.mkdir(parents=True, exist_ok=True)
        self.constant_path.mkdir(parents=True, exist_ok=True)
        self.system_path.mkdir(parents=True, exist_ok=True)
        self.zero_path.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Dictionary writers
    # ---------------------------------------------------------
    def write_controlDict(self, regions: bool = False) -> None:
        if regions:
            mesh_data = self.get_region_boundary_map()

        with open(self.system_path / "controlDict", "w") as f:
            f.write("FoamFile {\n")
            f.write("    version 2.0;\n")
            f.write("    format ascii;\n")
            f.write("    class dictionary;\n")
            f.write("    object controlDict;\n")
            f.write("}\n")
            f.write("\n")
            f.write("application     foamMultiRun;\n")
            f.write("\n")
            if regions:
                f.write("regionSolvers\n")
                f.write("{\n")
                for region in mesh_data:
                    f.write("   "+str(region)+"      solid;\n")

                f.write("}\n")

            f.write("startFrom       startTime;\n")
            f.write("startTime       0;\n")
            f.write("stopAt          endTime;\n")

            if self.time_settings["type"] == "steady":
                f.write("endTime         1;\n")
                f.write("deltaT          1;\n")
            else:
                dt = self.time_settings["time_step"]
                end = self.time_settings["end_time"]
                f.write("endTime         "+str(end)+";\n")
                f.write("deltaT          "+str(dt)+";\n")

            f.write("writeControl    timeStep;\n")
            f.write("writeInterval   1;\n")
            f.write("purgeWrite      0;\n")
            f.write("writeFormat     ascii;\n")
            f.write("writePrecision  6;\n")
            f.write("writeCompression off;\n")
            f.write("timeFormat      general;\n")
            f.write("timePrecision   6;\n")
            f.write("runTimeModifiable true;\n")
            f.write("maxCo           0.3;\n")
            f.write("maxDi           10.0;\n")
            f.write("adjustTimeStep  false;\n")

    def write_fvSchemes(self, regions: bool = False) -> None:
        if regions:
            mesh_data = self.get_region_boundary_map()
            for region in mesh_data:
                path_system_region = self.system_path / region
                path_system_region.mkdir(parents=True, exist_ok=True)
                with open(path_system_region / "fvSchemes", "w") as f:
                    f.write("FoamFile {\n")
                    f.write("    version 2.0;\n")
                    f.write("    format ascii;\n")
                    f.write("    class dictionary;\n")
                    f.write("    object fvSchemes;\n")
                    f.write("}\n")
                    f.write("\n")
                    f.write("ddtSchemes {\n")
                    f.write("    default Euler;\n")
                    f.write("}\n")
                    f.write("\n")
                    f.write("laplacianSchemes {\n")
                    f.write("    default Gauss linear limited corrected 0.33;\n")
                    f.write("}\n")
                    f.write("\n")
                    f.write("divSchemes {\n")
                    f.write("    default none;\n")
                    f.write("}\n")
                    f.write("\n")
                    f.write("gradSchemes {\n")
                    f.write("    default Gauss linear;\n")
                    f.write("}\n")
                    f.write("\n")
                    f.write("interpolationSchemes {\n")
                    f.write("    default linear;\n")
                    f.write("}\n")

            return None

        with open(self.system_path / "fvSchemes", "w") as f:
            f.write("FoamFile {\n")
            f.write("    version 2.0;\n")
            f.write("    format ascii;\n")
            f.write("    class dictionary;\n")
            f.write("    object fvSchemes;\n")
            f.write("}\n")
            f.write("\n")
            f.write("ddtSchemes {\n")
            f.write("    default Euler;\n")
            f.write("}\n")
            f.write("\n")
            f.write("laplacianSchemes {\n")
            f.write("    default Gauss linear limited corrected 0.33;\n")
            f.write("}\n")
            f.write("\n")
            f.write("divSchemes {\n")
            f.write("    default none;\n")
            f.write("}\n")
            f.write("\n")
            f.write("gradSchemes {\n")
            f.write("    default Gauss linear;\n")
            f.write("}\n")
            f.write("\n")
            f.write("interpolationSchemes {\n")
            f.write("    default linear;\n")
            f.write("}\n")

    def write_fvSolution(self, regions: bool = False) -> None:
        if regions:
            mesh_data = self.get_region_boundary_map()
            for region in mesh_data:
                path_system_region = self.system_path / region
                path_system_region.mkdir(parents=True, exist_ok=True)
                with open(path_system_region / "fvSolution", "w") as f:
                    f.write('/*--------------------------------*- C++ -*----------------------------------*\\ \n')
                    f.write('  =========                 |\n')
                    f.write('  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox\n')
                    f.write('   \\    /   O peration     | Website:  https://openfoam.org\n')
                    f.write('    \\  /    A nd           | Version:  11\n')
                    f.write('     \\/     M anipulation  |\n')
                    f.write('\\*---------------------------------------------------------------------------*/\n')
                    f.write('FoamFile\n')
                    f.write('{\n')
                    f.write('    format      ascii;\n')
                    f.write('    class       dictionary;\n')
                    f.write('    location    "system/'+str(region)+'";\n')
                    f.write('    object      fvSolution;\n')
                    f.write('}\n')
                    f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
                    f.write('\n')
                    f.write('solvers\n')
                    f.write('{\n')
                    f.write('    "(h|e)"\n')
                    f.write('    {\n')
                    f.write('        solver           PCG;\n')
                    f.write('	preconditioner   DIC;\n')
                    f.write('        smoother         symGaussSeidel;\n')
                    f.write('        tolerance        1e-6;\n')
                    f.write('        relTol           0.01;\n')
                    f.write('        nPreSweeps       2;\n')
                    f.write('        nPostSweeps      2;\n')
                    f.write('        nFinestSweeps    2;\n')
                    f.write('        cacheAgglomeration true;\n')
                    f.write('        agglomerator     faceAreaPair;\n')
                    f.write('        mergeLevels      1;\n')
                    f.write('    }\n')
                    f.write('\n')
                    f.write('    "(h|e)Final"\n')
                    f.write('    {\n')
                    f.write('        $h;\n')
                    f.write('        relTol           0;\n')
                    f.write('    }\n')
                    f.write('}\n')
                    f.write('\n')
                    f.write('PIMPLE\n')
                    f.write('{\n')
                    f.write('	nNonOrthogonalCorrectors 3;  // mesh has faces up to 89.7 degrees non-orthogonality\n')
                    f.write("   nOuterCorrectors            50;\n")
                    f.write('}\n')
                    f.write('\n')
                    f.write('\n')
                    f.write('// ************************************************************************* //\n')

            return None

        with open(self.system_path / "fvSolution", "w") as f:
            f.write('/*--------------------------------*- C++ -*----------------------------------*\\ \n')
            f.write('  =========                 |\n')
            f.write('  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox\n')
            f.write('   \\    /   O peration     | Website:  https://openfoam.org\n')
            f.write('    \\  /    A nd           | Version:  11\n')
            f.write('     \\/     M anipulation  |\n')
            f.write('\\*---------------------------------------------------------------------------*/\n')
            f.write('FoamFile\n')
            f.write('{\n')
            f.write('    format      ascii;\n')
            f.write('    class       dictionary;\n')
            f.write('    location    "system/heater";\n')
            f.write('    object      fvSolution;\n')
            f.write('}\n')
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
            f.write('\n')
            f.write('solvers\n')
            f.write('{\n')
            f.write('    "(h|e)"\n')
            f.write('    {\n')
            f.write('        solver           PCG;\n')
            f.write('	preconditioner   DIC;\n')
            f.write('        smoother         symGaussSeidel;\n')
            f.write('        tolerance        0.01;\n')
            f.write('        relTol           0.05;\n')
            f.write('        nPreSweeps       2;\n')
            f.write('        nPostSweeps      2;\n')
            f.write('        nFinestSweeps    2;\n')
            f.write('        cacheAgglomeration true;\n')
            f.write('        agglomerator     faceAreaPair;\n')
            f.write('        mergeLevels      1;\n')
            f.write('    }\n')
            f.write('\n')
            f.write('    "(h|e)Final"\n')
            f.write('    {\n')
            f.write('        $h;\n')
            f.write('        relTol           0;\n')
            f.write('    }\n')
            f.write('}\n')
            f.write('\n')
            f.write('PIMPLE\n')
            f.write('{\n')
            f.write('	nNonOrthogonalCorrectors 1;  // mesh has faces up to 89.7 degrees non-orthogonality\n')
            f.write('}\n')
            f.write('\n')
            f.write('\n')
            f.write('// ************************************************************************* //\n')

    def write_regionProperties(self) -> None:
        with open(self.constant_path / "regionProperties", "w") as f:
                f.write("regions\n")
                f.write("(\n")
                f.write("    solid\n")
                f.write(");\n")


    # ---------------------------------------------------------
    # Heat flux / boundary data
    # ---------------------------------------------------------
    def write_heat_flux(self, xyz: np.ndarray, heat_flux: np.ndarray) -> None:
        """Write ``constant/data.txt`` (x y z q per line) read by the coded plasma BC."""
        data = np.column_stack([xyz, heat_flux])
        np.savetxt(self.constant_path / "data.txt",
                   data, header="time: 0.0", comments="")

    # ---------------------------------------------------------
    # High-level build function
    # ---------------------------------------------------------
    def build_case(self, xyz: np.ndarray, heat_flux: np.ndarray) -> None:
        """Write the region-independent part of the case (before the mesh is split)."""
        self.create_directory_structure()
        self.write_controlDict()
        self.write_fvSchemes()
        self.write_fvSolution()
        self.write_regionProperties()
        self.write_heat_flux(xyz, heat_flux)

    def write_T(self) -> None:
        mesh_data = self.get_region_boundary_map()
        for region in mesh_data:
            with open(self.zero_path / region / "T", "w", encoding="utf-8") as f:
                f.write("FoamFile\n")
                f.write("{\n")
                f.write("    version     2.0;\n")
                f.write("    format      ascii;\n")
                f.write("    class       volScalarField;\n")
                f.write("    object      T;\n")
                f.write("}\n")
                f.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n")
                f.write("\n")
                f.write("dimensions      [0 0 0 1 0 0 0];\n")
                f.write("\n")
                f.write("internalField   uniform "+str(self.initial_temperature)+";\n")
                f.write("\n")
                f.write("boundaryField\n")
                f.write("{\n")

                for boundary in mesh_data[region]:
                    #print(boundary, region)
                    if region+"_to_" in boundary:
                        #print("Coupled boundary detected: "+boundary)
                        f.write("    "+boundary+"\n")
                        f.write("    {\n")
                        f.write("        type            coupledTemperature;\n")
                        f.write("        value           uniform "+str(self.initial_temperature)+";\n")
                        f.write("    }\n")
                        f.write("\n")
                        continue

                    if boundary == self.plasma_bc_name:
                        f.write("       "+boundary+"\n")
                        f.write("{\n")
                        write_openfoam_plasma_bc(f, self.initial_temperature)

                    else:
                        if boundary != "defaultFaces":
                            data_b = self.boundary_conditions[boundary]
                        elif boundary == "defaultFaces":
                            if boundary not in self.boundary_conditions.keys():
                                data_b = {"type": "heat_flux",
                                          "value_type": "constant",
                                          "value": "zeroGradient"}
                            else:
                                data_b = self.boundary_conditions[boundary]

                        f.write("    "+boundary+"\n")
                        f.write("    {\n")

                        if data_b["type"] == "htc":
                            f.write("        type            externalWallHeatFluxTemperature;\n")
                            f.write("        mode            flux;\n")
                            f.write("        kappaMethod     solidThermo;\n")
                            if data_b["value_type"] ==  "constant":
                                f.write("        h               uniform "+str(data_b["value"])+";\n")
                                f.write("        Ta              uniform "+str(data_b["T_ext"])+";\n")
                                f.write("        value           uniform "+str(self.initial_temperature)+";\n")
                            f.write("   }\n")
                            f.write("\n")

                        elif data_b["type"] == "heat_flux" and data_b["value"] != "zeroGradient":
                            f.write("        type            externalWallHeatFluxTemperature;\n")
                            f.write("        mode            flux;\n")
                            f.write("        kappaMethod     solidThermo;\n")
                            if data_b["value_type"] ==  "constant":
                                f.write("        q               uniform "+str(data_b["value"])+";\n")
                                f.write("        value           uniform "+str(self.initial_temperature)+";\n")
                            f.write("   }\n")
                            f.write("\n")

                        elif data_b["value"] == "zeroGradient":
                            f.write("    type            zeroGradient;\n")
                            f.write("   }\n")
                            f.write("\n")

                        elif data_b["type"] == "temperature":
                            f.write("    type            fixedValue;\n")
                            f.write("    value           uniform "+str(data_b["value"])+";\n")
                            f.write("   }\n")
                            f.write("\n")


                f.write("}\n")
                f.write("// ************************************************************************* //\n")




    def write_physicalProperties(self, regions: bool = False) -> None:
        if regions:
            mesh_data = self.get_region_boundary_map()

            for region in mesh_data:

                if self.material_properties:
                    mat_region = self.material_properties[region]

                    # Get density #############################################
                    density = mat_region.get("density", None)
                    density_type = density.get("type", None)
                    density_value = density.get("value", None)

                    if density_type == "constant":
                        rho_type_to_write = "rhoConst"
                        rho_value_to_write = "        rho             "+str(density_value)+";\n"
                    elif density_type == "table":
                        rho_type_to_write = "eIcoTabulated"
                        rho_value_to_write = "         rho\n"
                        rho_value_to_write += "{"
                        rho_value_to_write += "    values\n"
                        rho_value_to_write += "    (\n"
                        for vals in density_value:
                            rho_value_to_write += "            ("+str(vals[0])+" "+str(vals[1])+")\n"
                        rho_value_to_write += "    )\n"
                        rho_value_to_write += "}\n"
                    elif density_type == "polynomial":
                        rho_type_to_write = "ePolynomial"
                        rho_value_to_write = "rhoCoeffs<8>     ("
                        for val in density_value:
                            rho_value_to_write += " "+str(val)
                        rho_value_to_write += ");"

                    # Get heat capacity #######################################
                    heat_capacity = mat_region.get("heat_capacity", None)
                    heat_capacity_type = heat_capacity.get("type", None)
                    heat_capacity_value = heat_capacity.get("value", None)

                    if heat_capacity_type == "constant":
                        Cv_type_to_write = "eConst"
                        Cv_value_to_write = "        Cv             "+str(heat_capacity_value)+";\n"
                    elif heat_capacity_type == "table":
                        Cv_type_to_write = "eIcoTabulated"
                        Cv_value_to_write = "         Cv\n"
                        Cv_value_to_write += "{"
                        Cv_value_to_write += "    values\n"
                        Cv_value_to_write += "    (\n"
                        for vals in heat_capacity_value:
                            Cv_value_to_write += "            ("+str(vals[0])+" "+str(vals[1])+")\n"
                        Cv_value_to_write += "    )\n"
                        Cv_value_to_write += "}\n"
                    elif heat_capacity_type == "polynomial":
                        Cv_type_to_write = "hPolynomial"
                        Cv_value_to_write = "CvCoeffs<8>     ("
                        for val in heat_capacity_value:
                            Cv_value_to_write += " "+str(val)
                        Cv_value_to_write += ");"


                    # Get heat conductivity ###################################
                    heat_conductivity = mat_region.get("heat_conductivity", None)
                    heat_conductivity_type = heat_conductivity.get("type", None)
                    heat_conductivity_value = heat_conductivity.get("value", None)

                    if heat_conductivity_type == "constant":
                        kappa_type_to_write = "constIsoSolid"
                        kappa_value_to_write = "        kappa             "+str(heat_conductivity_value)+";\n"
                    elif heat_conductivity_type == "table":
                        kappa_type_to_write = "tabulatedSolidTransport"
                        kappa_value_to_write = "         kappa\n"
                        kappa_value_to_write += "{"
                        kappa_value_to_write += "    values\n"
                        kappa_value_to_write += "    (\n"
                        for vals in heat_conductivity_value:
                            kappa_value_to_write += "            ("+str(vals[0])+" "+str(vals[1])+")\n"
                        kappa_value_to_write += "    )\n"
                        kappa_value_to_write += "}\n"
                    elif heat_conductivity_type in ("polynomial", "polynomialSolid"):
                        kappa_type_to_write = "hPolynomial"
                        kappa_value_to_write = "kappaCoeffs<8>     ("
                        for val in heat_conductivity_value:
                            kappa_value_to_write += " "+str(val)
                        kappa_value_to_write += ");"

                else:
                    heat_conductivity = None
                    density = None
                    heat_capacity = None

                with open(self.constant_path / region / "physicalProperties", "w") as f:
                    f.write("FoamFile\n")
                    f.write("{\n")
                    f.write("    format      ascii;\n")
                    f.write("    class       dictionary;\n")
                    f.write('    location    "constant/'+str(region)+'";\n')
                    f.write("    object      physicalProperties;\n")
                    f.write("}\n")
                    f.write("\n")
                    f.write("thermoType\n")
                    f.write("{\n")
                    f.write("    type            heSolidThermo;\n")
                    f.write("    mixture         pureMixture;\n")

                    # Write heat conductivity type ############################
                    if heat_conductivity:
                        f.write("    transport       "+kappa_type_to_write+";\n")
                    else:
                        f.write("    transport       polynomialSolid;\n")

                    if heat_capacity:
                        f.write("    thermo       "+Cv_type_to_write+";\n")
                    else:
                        f.write("    thermo          eConst;\n")

                    if density:
                        f.write("    equationOfState       "+rho_type_to_write+";\n")
                    else:
                        f.write("    equationOfState rhoConst;\n")

                    f.write("    specie          specie;\n")
                    f.write("    energy          sensibleInternalEnergy;\n")
                    f.write("}\n")
                    f.write("\n")
                    f.write("mixture\n")
                    f.write("{\n")
                    f.write("    specie\n")
                    f.write("    {\n")
                    f.write("        molWeight       44.09; // SiO\n")
                    f.write("    }\n")
                    f.write("\n")
                    f.write("    equationOfState\n")
                    f.write("    {\n")

                    if density:
                        f.write(rho_value_to_write)
                    else:
                        f.write("        rho             3060;// kg/m3\n")

                    f.write("    }\n")
                    f.write("\n")
                    f.write("    thermodynamics\n")
                    f.write("    {\n")
                    f.write("        Hf              0;\n")

                    if heat_capacity:
                        f.write(Cv_value_to_write)
                    else:
                        f.write("        Cv              900;\n")

                    f.write("    }\n")
                    f.write("\n")
                    f.write("    transport\n")
                    f.write("    {\n")

                    if heat_conductivity:
                        f.write(kappa_value_to_write)
                    else:
                        f.write("        kappaCoeffs<8>  (  1.82e+01  -7.40e-03   1.20e-05  -8.50e-09   0 0 0 0 );\n")
                    f.write("    }\n")
                    f.write("}\n")
