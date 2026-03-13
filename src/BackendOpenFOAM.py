import os
import re
import subprocess
import shlex
import pdb
from pathlib import Path
import numpy as np
from OpenFOAMCaseBuilder import OpenFOAMCaseBuilder

# Connector between IR workflow - thermal model and OpenFOAM. 
class BackendOpenFOAM(object):
    """Documentation for BackendOpenFOAM

    """
    def __init__(self, mesh_file, mesh_type, plasma_bc_name, root_dir, time_stepping, openfoam_bashrc_dir, openfoam_case_name=None, initial_temperature=None, boundary_conditions=None):
        super(BackendOpenFOAM, self).__init__()
        self.mesh_file = mesh_file
        self.mesh_type = mesh_type
        self.plasma_bc_name = plasma_bc_name
        self.root_dir = root_dir
        self.time_stepping = time_stepping
        self.openfoam_bashrc_dir = openfoam_bashrc_dir
        self.openfoam_case_name = openfoam_case_name
        self.initial_temperature = initial_temperature
        self.boundary_conditions = boundary_conditions
        
        if time_stepping["type"].lower() != "steady":
            self.is_transient = True
            self.time_step = time_stepping["time_step"]
            self.end_time = time_stepping["end_time"]
        else:
            self.is_transient = False
            self.time_step = None
            self.end_time = None


    def get_irflow_pass_data(self, irflow_pass_data):
        self.irflow_pass_data = irflow_pass_data

    def run(self, irflow_case_dir):
        path = Path(irflow_case_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {path}")

        #path = Path(irflow_case_dir, "ThermalModel")
        #if not path.is_dir():
        #    raise FileNotFoundError(f"Directory does not exist: {path}")

        # Run OpenFOAM case here using subprocess and the OpenFOAM bashrc
        # source the OpenFOAM bashrc to set up the environment

        xyz = self.irflow_pass_data["xyz"]
        heat_flux = self.irflow_pass_data["heat_flux"]


        of_hf_data = np.zeros((xyz.shape[0], 4))
        of_hf_data[:, 0:3] = xyz
        of_hf_data[:, 3] = heat_flux

        if self.openfoam_case_name is None:
            self.openfoam_case_name = "openfoam_case"

        builder = OpenFOAMCaseBuilder(
            case_root=path,
            case_name=self.openfoam_case_name,
            time_settings=self.time_stepping,
            mesh_type=self.mesh_type,
            plasma_bc_name=self.plasma_bc_name,
            initial_temperature=self.initial_temperature,
            boundary_conditions=self.boundary_conditions
        )

        builder.build_case(xyz, heat_flux)

        #__import__("pdb").set_trace()
        of_case_path = Path(path, self.openfoam_case_name)
        of_case_path.mkdir(parents=True, exist_ok=True)
        #of_case_path.mkdir(parents=True, exist_ok=True)

        #of_case_constant_path = Path(of_case_path, "constant")
        #of_case_constant_path.mkdir(parents=True, exist_ok=True)

        #np.savetxt(Path(of_case_constant_path, "heat_flux.txt"),
        #           of_hf_data, header="time: 0.0", comments="")

        #of_case_system_path = Path(of_case_path, "system")
        #of_case_system_path.mkdir(parents=True, exist_ok=True)

        # Build the shell script that will run inside bash

        from templates import write_openfoam_sh_script
        script_prepare_path = Path(of_case_path, "prepare.sh")
        script_run_path = Path(of_case_path, "run.sh")

        script_prepare = open(Path(of_case_path, "prepare.sh"), "w")
        write_openfoam_sh_script(script_prepare, of_case_path, self.openfoam_bashrc_dir, self.mesh_file)
        script_prepare.write("ideasUnvToFoam "+shlex.quote(self.mesh_file)+" -case "+shlex.quote(str(of_case_path))+"\n")
        script_prepare.write("splitMeshRegions -cellZones -overwrite\n")
        script_prepare.close()
        Path(of_case_path, "prepare.sh").chmod(0o755)
        
        
        # Execute inside an interactive bash shell
        result = subprocess.run(
            [str(script_prepare_path)],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL
        )

        print(result.stdout)
        print(result.stderr)

        builder.write_T()
        builder.write_controlDict(regions=True)
        builder.write_physicalProperties(regions=True)
        builder.write_fvSchemes(regions=True)
        builder.write_fvSolution(regions=True)
        
        # Execute inside an interactive bash shell
        script_run = open(Path(of_case_path, "run.sh"), "w")
        write_openfoam_sh_script(script_run, of_case_path, self.openfoam_bashrc_dir, self.mesh_file)
        script_run.write("foamMultiRun\n")

        mesh_data = builder.get_region_boundary_map()
        region_plasma_bc_name = None
        for region in mesh_data:
            if self.plasma_bc_name in mesh_data[region]:
                region_plasma_bc_name = region
                mesh_data[region].remove(self.plasma_bc_name)
                exclude_regions = "'("
                for patch in mesh_data[region]:
                    exclude_regions += patch + " "

                exclude_regions = exclude_regions[:-1] + ")'"
   
                script_run.write(f"foamMultiRun\n")
                script_run.write(f"foamToVTK -region {region} -excludePatches "+exclude_regions+"\n")
                break
        
        script_run.close()
        
        Path(of_case_path, "run.sh").chmod(0o755)

        result = subprocess.run(
            [str(script_run_path)],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL
        )

        # Create 'foam' file
        with open('case.foam', 'w') as fp:
            pass


        print(result.stdout)
        print(result.stderr)

        pdb.set_trace()
        #foamToVTK -region steel_up -excludePatches '(const_T steel_up_to_steel_down)'
        #pdb.set_trace()
        import pyvista as pv

        # Get only last timestep
        pvmesh = pv.read(Path(of_case_path, "VTK",
                              region_plasma_bc_name,
                              self.plasma_bc_name,
                              self.plasma_bc_name+"_"+ \
                              str(int(self.end_time/self.time_step))+".vtk"))

        # cell based temperature is stored in "T" array in the VTK file
        T = pvmesh["T"]                     # cell-based temperature
        cell_centers = pvmesh.cell_centers()
        centers_coords = cell_centers.points      # coordinates of cell centers

        # node based temperature is stored in points "T" array in the VTK file
        T_nodes = pvmesh.point_data["T"]
        nodes = pvmesh.points
