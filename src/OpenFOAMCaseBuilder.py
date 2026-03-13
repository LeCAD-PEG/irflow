from pathlib import Path
import numpy as np
import pdb

class OpenFOAMCaseBuilder:
    def __init__(self, case_root, case_name, time_settings, mesh_type, plasma_bc_name, initial_temperature=300.0, boundary_conditions=None):
        self.case_root = Path(case_root)
        self.case_name = case_name
        self.time_settings = time_settings
        self.mesh_type = mesh_type
        self.plasma_bc_name = plasma_bc_name
        self.initial_temperature = initial_temperature
        self.boundary_conditions = boundary_conditions or {}

        self.case_path = self.case_root / self.case_name
        self.constant_path = self.case_path / "constant"
        self.system_path = self.case_path / "system"
        self.zero_path = self.case_path / "0"


    def get_region_boundary_map(self, ):
        region_map = {}
        exclude = {"polyMesh", "regionProperties"}
    
        for region_dir in self.constant_path.iterdir():
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
    def create_directory_structure(self):
        self.case_path.mkdir(parents=True, exist_ok=True)
        self.constant_path.mkdir(parents=True, exist_ok=True)
        self.system_path.mkdir(parents=True, exist_ok=True)
        self.zero_path.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Dictionary writers
    # ---------------------------------------------------------
    def write_controlDict(self, regions=False):
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

    def write_fvSchemes(self, regions=False):
        if regions:
            mesh_data = self.get_region_boundary_map()
            pdb.set_trace()
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
                    f.write("    default Gauss linear corrected;\n")
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
            f.write("    default Gauss linear corrected;\n")
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

    def write_fvSolution(self, regions=False):
        if regions:
            mesh_data = self.get_region_boundary_map()
            for region in mesh_data:
                path_system_region = self.system_path / region
                path_system_region.mkdir(parents=True, exist_ok=True)
                with open(path_system_region / "fvSolution", "w") as f:
                    f.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
                    f.write('  =========                 |\n')
                    f.write('  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox\n')
                    f.write('   \\    /   O peration     | Website:  https://openfoam.org\n')
                    f.write('    \\  /    A nd           | Version:  11\n')
                    f.write('     \\/     M anipulation  |\n')
                    f.write('\*---------------------------------------------------------------------------*/\n')
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
                    f.write('	nNonOrthogonalCorrectors 1;\n')
                    f.write('}\n')
                    f.write('\n')
                    f.write('\n')
                    f.write('// ************************************************************************* //\n')

            return None
            
        with open(self.system_path / "fvSolution", "w") as f:
            f.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
            f.write('  =========                 |\n')
            f.write('  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox\n')
            f.write('   \\    /   O peration     | Website:  https://openfoam.org\n')
            f.write('    \\  /    A nd           | Version:  11\n')
            f.write('     \\/     M anipulation  |\n')
            f.write('\*---------------------------------------------------------------------------*/\n')
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
            f.write('	nNonOrthogonalCorrectors 1;\n')
            f.write('}\n')
            f.write('\n')
            f.write('\n')
            f.write('// ************************************************************************* //\n')

    def write_regionProperties(self):
        with open(self.constant_path / "regionProperties", "w") as f:
                f.write("regions\n")
                f.write("(\n")
                f.write("    solid\n")
                f.write(");\n")


    # ---------------------------------------------------------
    # Heat flux / boundary data
    # ---------------------------------------------------------
    def write_heat_flux(self, xyz, heat_flux):
        data = np.column_stack([xyz, heat_flux])
        np.savetxt(self.constant_path / "data.txt",
                   data, header="time: 0.0", comments="")

    # ---------------------------------------------------------
    # High-level build function
    # ---------------------------------------------------------
    def build_case(self, xyz, heat_flux):
        self.create_directory_structure()
        self.write_controlDict()
        self.write_fvSchemes()
        self.write_fvSolution()
        self.write_regionProperties()
        self.write_heat_flux(xyz, heat_flux)

    def write_T(self):
        mesh_data = self.get_region_boundary_map()
        for region in mesh_data:
            with open(self.zero_path / region / "T", "w") as f:
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
                    if boundary == self.plasma_bc_name:
                        from templates import write_openfoam_plasma_bc
                        write_openfoam_plasma_bc(f, self.initial_temperature)


                f.write("}\n")
                f.write("// ************************************************************************* //\n")




    def write_physicalProperties(self, regions=False):
        if regions:
            mesh_data = self.get_region_boundary_map()
        
            for region in mesh_data:
                with open(self.constant_path / region / "physicalProperties", "w") as f:
                    f.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
                    f.write('  =========                 |\n')
                    f.write('  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox\n')
                    f.write('   \\    /   O peration     | Website:  https://openfoam.org\n')
                    f.write('    \\  /    A nd           | Version:  11\n')
                    f.write('     \\/     M anipulation  |\n')
                    f.write('\*---------------------------------------------------------------------------*/\n')
                    f.write('FoamFile\n')
                    f.write('{\n')
                    f.write('    format      ascii;\n')
                    f.write('    class       dictionary;\n')
                    f.write('    location    "constant/metal";\n')
                    f.write('    object      physicalProperties;\n')
                    f.write('}\n')
                    f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
                    f.write('\n')
                    f.write('thermoType\n')
                    f.write('{\n')
                    f.write('    type            heSolidThermo;\n')
                    f.write('    mixture         pureMixture;\n')
                    f.write('    transport       constIsoSolid;\n')
                    f.write('    thermo          eConst;\n')
                    f.write('    equationOfState rhoConst;\n')
                    f.write('    specie          specie;\n')
                    f.write('    energy          sensibleInternalEnergy;\n')
                    f.write('}\n')
                    f.write('\n')
                    f.write('mixture\n')
                    f.write('{\n')
                    f.write('    // SiO\n')
                    f.write('\n')
                    f.write('    specie\n')
                    f.write('    {\n')
                    f.write('        molWeight       184;\n')
                    f.write('    }\n')
                    f.write('\n')
                    f.write('    equationOfState\n')
                    f.write('    {\n')
                    f.write('        rho             3084;   // kg/m3 at 400°C\n')
                    f.write('    }\n')
                    f.write('\n')
                    f.write('    transport\n')
                    f.write('    {\n')
                    f.write('        kappa           15.25;  // W/(m·K) at 400°C\n')
                    f.write('    }\n')
                    f.write('\n')
                    f.write('    thermodynamics\n')
                    f.write('    {\n')
                    f.write('        Hf              0;\n')
                    f.write('        Cv              1084;   // J/(kg·K) at 400°C\n')
                    f.write('    }\n')
                    f.write('}\n')
                    f.write('\n')
                    f.write('\n')
                    f.write('// ************************************************************************* //\n')
                    f.write('\n')
            
    
