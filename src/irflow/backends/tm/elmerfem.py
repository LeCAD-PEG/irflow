"""Thermal model backend for ElmerFEM (sif template + Elmer mesh, run with ElmerSolver)."""

from __future__ import annotations

import copy
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from scipy.interpolate import NearestNDInterpolator

from irflow.backends.base import ThermalModelBackend
from irflow.code_paths import get_code_path
from irflow.data import HeatFluxField, SurfaceMesh, TemperatureField
from irflow.io.vtk import to_polydata
from irflow.mapping import check_overlap


def elmer_executable(name: str) -> str:
    """Return ElmerGrid/ElmerSolver from ``[elmerfem] bin_dir`` in code paths, else PATH."""
    bin_dir = get_code_path("elmerfem", "bin_dir")
    return str(Path(bin_dir) / name) if bin_dir else name


def get_boundary_conditions(sif_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    boundary_conditions = {}
    for key in sif_data.keys():
        if "Boundary Condition" in key:

            sif_bc_id = int(key.split()[-1].replace(" ", ""))-1
            sif_group_2d_id = None
            sif_temperature = None
            sif_heat_flux = None
            sif_external_temperature = None
            sif_htc = None

            for ii in sif_data[key]:
                if "Target boundaries" in ii:
                    sif_group_2d_id = int(sif_data[key][ii].replace(" ", ""))-1
                elif "external temperature" in ii.lower():
                    sif_external_temperature = sif_data[key][ii].replace(" ", "")
                elif "temperature" in ii.lower():
                    sif_temperature = sif_data[key][ii].replace(" ", "")
                elif "heat flux" in ii.lower():
                    sif_heat_flux = sif_data[key][ii].replace(" ", "")
                elif "heat transfer coefficient" in ii.lower():
                    sif_htc = sif_data[key][ii].replace(" ", "")

            # TODO: check that boundary conditions make sense (i.e. no
            # temperature and heat flux on the same surface)
            #if None not in [temperature, heat_flux, htc]:
            #etc.
            
            # IR flow format dict
            irflow_key = "bc_id_"+str(sif_bc_id)
            bc = {"group_2d_id": sif_group_2d_id}
            if sif_temperature:
                bc["type"] = "temperature"
                bc["value"] = sif_temperature
            elif sif_heat_flux:
                bc["type"] = "heat_flux"
                bc["value"] = sif_heat_flux
            elif sif_htc:
                bc["type"] = "htc"
                bc["value"] = sif_htc
                bc["external_value"] = sif_external_temperature

            bc["value_type"] = "const"
            boundary_conditions[irflow_key] = bc

    return boundary_conditions

def get_material_properties(sif_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sif_materials = {}
    sif_bodies = {}

    for key in sif_data.keys():
        if "Body" in key:
            __body_id = None
            __body_material_id = None
            for ii in sif_data[key]:
                if "Target Bodies" in ii:
                    __body_id = int(sif_data[key][ii].replace(" ", ""))-1
                elif "Name" in ii:
                    body_name = sif_data[key][ii].replace(" ", "")
                elif "Material" in ii:
                    __body_material_id = int(sif_data[key][ii].replace(" ", ""))-1

            if None in [__body_id, __body_material_id]:
                # Throw exception
                raise ValueError( f"Some of body properties in Body '{__body_id}' are not defined."
                                  "\nBody ID'{__body_id}'"
                                  "\nBody Material ID'{__body_material_id}'.")

            key_split = key.split()
            new_key = key_split[0] + " " + str(int(key_split[1].replace(" ", ""))-1)
            sif_bodies[key] = {"Target Bodies": __body_id, "Material": __body_material_id}
            
        elif "Material" in key:
            material_id = int(key.split()[1].replace(" ", ""))-1
            __density = None
            __capacity = None
            __conductivity = None
            __material_name = None

            for ii in sif_data[key]:

                if "Density" in ii:
                    if type(sif_data[key][ii]) == np.ndarray:
                        __density = sif_data[key][ii]
                    else:
                        __density = sif_data[key][ii].replace(" ", "")
                elif "Name" in ii:
                    __material_name = sif_data[key][ii].replace(" ", "")
                    if "!" in __material_name:
                        __material_name = __material_name.split('!', 1)[0].rstrip().replace(" ", "")
                elif "Heat Capacity" in ii:
                    if type(sif_data[key][ii]) == np.ndarray:
                        __capacity = sif_data[key][ii] 
                    else:
                        __capacity = sif_data[key][ii].replace(" ", "")
                elif "Heat Conductivity" in ii:
                    if type(sif_data[key][ii]) == np.ndarray:
                        __conductivity = sif_data[key][ii] 
                    else:
                        __conductivity = sif_data[key][ii].replace(" ", "")

            if __density is None or __capacity is None or __conductivity is None:
                # Throw exception
                raise ValueError( f"Some of material properties in Material '{material_id}' are not defined."
                                  "\nCapacity'{__capacity}'"
                                  "\nConductivity'{__conductivity}'"
                                  "\nDensity'{__density}'.")

            # Decrease Material ID by 1 to convert to 0-based indexing
            # for internal use, will be converted back to 1-based when
            # writing sif file
            key_split = key.split()
            new_key = key_split[0] + " " + str(int(key_split[1].replace(" ", ""))-1)
            
            sif_materials[new_key] = {"material_id": material_id,
                                  "Density": __density,
                                  "Name": __material_name,
                                  "Heat Capacity": __capacity,
                                  "Heat Conductivity": __conductivity}

    # Merge data from sif_bodies and sif_materials into one dict
    material_properties = {}
    for key in sif_bodies:
        group_3d_id = sif_bodies[key]["Target Bodies"]
        material_id = sif_bodies[key]["Material"]

        sif_material_key = "Material "+str(material_id)
        try:
            sif_material = sif_materials[sif_material_key]
        except KeyError:
            raise ValueError( f"Material with ID '{material_id}' does not exist."
                                  "Check Body '{}'"
                                  "\nCapacity'{__capacity}'"
                                  "\nConductivity'{__conductivity}'"
                                  "\nDensity'{__density}'.")

        # IR flow format
        material_capacity = sif_material["Heat Capacity"]
        material_density = sif_material["Density"]
        material_conductivity = sif_material["Heat Conductivity"]
        material_name = sif_material["Name"]
        material_key = "mat_id_" + str(material_id)
        properties = {"heat_capacity": {"type":"const", "values": material_capacity},
                      "density": {"type":"const", "values": material_density},
                      "heat_conductivity": {"type":"const", "values": material_conductivity}}
        material_properties[material_key] = {"material_name":material_name,
                                             "group_3d_id": group_3d_id,
                                             "properties": properties}

    return material_properties

def is_new_block(line: str) -> bool:
    """return True if we are at start of block"""
    if line[:6].lower() == "header" or  \
       line[:10].lower() == "simulation" or  \
       line[:9].lower() == "constants" or  \
       line[:4].lower() == "body" or  \
       line[:6].lower() == "solver" or  \
       line[:8].lower() == "material" or  \
       line[:8].lower() == "equation" or  \
       line[:8].lower() == "boundary":
        return True
    else:
        return False


# Connector between IR workflow - thermal model and Elmer
class BackendElmerFEM(ThermalModelBackend):
    """Connector between the irflow thermal model step and ElmerFEM."""

    def __init__(
        self,
        mesh_file: str,
        mesh_type: str,
        sif_file: str,
        plasma_bc_id: int,
        time_stepping: dict[str, Any],
    ) -> None:
        self.mesh_file = mesh_file
        self.sif_file = sif_file
        self.mesh_type = mesh_type
        self.plasma_bc_id = plasma_bc_id - 1 # convert to 0-based
                                             # indexing for internal
                                             # use, will be converted
                                             # back to 1-based when
                                             # writing mesh and sif
                                             # file
        self.case_dir: Path | None = None

        self.elmer_mesh_nodes = None # os.path.join(self.mesh_file, "mesh.nodes")
        self.elmer_mesh_elements = None # os.path.join(self.mesh_file, "mesh.elements")
        self.elmer_mesh_boundary = None # os.path.join(self.mesh_file, "mesh.boundary")

        if time_stepping["type"].lower() != "steady":
            self.is_transient = True
            self.time_step = time_stepping["time_step"]
            self.end_time = time_stepping["end_time"]
        else:
            self.is_transient = False
            self.time_step = None
            self.end_time = None

        # Filled in run(): sif and mesh are read and converted in the step directory
        self.sif_data: dict[str, Any] = {}
        self.mesh: dict[str, Any] = {}
        self.material_properties: dict[str, Any] = {}
        self.boundary_conditions: dict[str, Any] = {}
        self.plasma_triangles: np.ndarray | None = None  # (M, 3) Elmer node indices
        self.result: TemperatureField | None = None

    def get_elmer_mesh(
        self, mesh_type: str, mesh_file: str | Path, dir_path: str | Path | None = None
    ) -> None:
        mesh_file = Path(mesh_file)

        if not dir_path:
            dir_path = "./"
    
        if mesh_type.lower() == "mesh":
            self.elmer_mesh_nodes = np.loadtxt(mesh_file / "mesh.nodes").astype(float)
            self.elmer_mesh_nodes[:, 0] -= 1 
            self.elmer_mesh_elements = np.loadtxt(mesh_file / "mesh.elements").astype(int)-1
            self.elmer_mesh_boundary = np.loadtxt(mesh_file / "mesh.boundary").astype(int)-1
    
        elif mesh_type.lower() == "unv":
            # 1) Convert UNV to Elmer mesh
            #    ElmerGrid 8 2 input.unv -out mesh_dir
            #out_dir =  mesh_file.parent / "elmer_mesh"
            dir_path = Path(dir_path)

            dir_path.mkdir(parents=True, exist_ok=True)

            elmer_mesh_dir = dir_path / "elmer_mesh"
            
            print("Running ElmerGrid to convert UNV → Elmer mesh ...")
            print("Directory for Elmer mesh:", elmer_mesh_dir)
            result = subprocess.run(
                [elmer_executable("ElmerGrid"), "8", "2", str(mesh_file), "-out", str(elmer_mesh_dir), "-autoclean"],
                capture_output=True,
                text=True
            )
    
            print(result.stdout)
            print(result.stderr)
    
            if result.returncode != 0:
                raise RuntimeError("ElmerGrid failed with return code {}".format(result.returncode))
    
            # 2) Preberi generirane Elmer mesh datoteke
            #elmer_mesh_dir = dir_path / "elmer_mesh"
            print("Directory for Elmer mesh:", elmer_mesh_dir)
            self.elmer_mesh_nodes = np.loadtxt(elmer_mesh_dir / "mesh.nodes").astype(float)
            self.elmer_mesh_nodes[:, 0] -= 1
            self.elmer_mesh_elements = np.loadtxt(elmer_mesh_dir / "mesh.elements").astype(int)-1
            self.elmer_mesh_boundary = np.loadtxt(elmer_mesh_dir / "mesh.boundary").astype(int)-1
    
        else:
            raise ValueError(f"Unknown mesh type: {mesh_type}. Only" \
            "'mesh' (Elmer FEM mesh) and 'unv' (Universal) are" \
            "supported.")


    def get_tm_mesh(self) -> dict[str, Any]:
        #if self.mesh_type.lower() == "mesh":
        nodes = self.elmer_mesh_nodes[:, -3:]

        elements = self.elmer_mesh_elements.astype("int")
        cells_3d = elements[:, -4:]
        groups_3d = {}
        for element in enumerate(elements[:, 1]):
            iterator = element[0]
            element_id = element[1]
            if "group_3d_id_"+str(element_id) not in groups_3d.keys():
                groups_3d["group_3d_id_"+str(element_id)] = [iterator]
            else:
                groups_3d["group_3d_id_"+str(element_id)].append(iterator)

        for key in groups_3d.keys():
            groups_3d[key] = np.array(groups_3d[key])

        boundaries = self.elmer_mesh_boundary.astype("int")
        cells_2d = boundaries[:, -3:]
        groups_2d = {}
        for boundary in enumerate(boundaries[:, 1]):
            iterator = boundary[0]
            boundary_id = boundary[1]
            if "group_2d_id_"+str(boundary_id) not in groups_2d.keys():
                groups_2d["group_2d_id_"+str(boundary_id)] = [iterator]
            else:
                groups_2d["group_2d_id_"+str(boundary_id)].append(iterator)

        for key in groups_2d.keys():
            groups_2d[key] = np.array(groups_2d[key])

        self.mesh = {"nodes": nodes,
                     "cells_3d": cells_3d,
                     "cells_2d": cells_2d,
                     "groups_3d": groups_3d,
                     "groups_2d": groups_2d}

        return self.mesh

    def read_sif(self, filename: str | Path | None = None) -> dict[str, Any]:
        if not filename:
            filename = self.sif_file


        sif_data = {}

        tabular_data = []

        with open(filename, "r") as f:
            f_content = f.readlines()#.split("\n")

        is_table = False
        for line in f_content: 
            line = line.replace("\n", "") 

            if "real" in line.lower():
                is_table = True
                continue

            if "end" in line.lower():
                if is_table:
                    new_block[key] = np.array(tabular_data)
    
                    is_table = False
                else:
                    sif_data[new_block_name] = new_block

                continue


            if is_table:
                T, variable = line.split() # remove comments from line
                tabular_data.append([float(T), float(variable)])
                continue

            if is_new_block(line):
                new_block_name = line
                new_block = {}
            #elif line[:3].lower() == "end":
            #    # end of block
            #    sif_data[new_block_name] = new_block
            #   continue
            elif line == "" or line.lstrip()[0] == "!":
                continue
            else:
                line = line.split("!")[0] # remove comments from line
                if new_block_name == "Header":
                    # Header block does not have '=' separator
                    line = line.lstrip()
                    if "Mesh DB" in line:
                        key = "Mesh DB"
                        value = line.replace(key, "")
                        new_block[key] = value
                    elif "Include Path" in line:
                        key = "Include Path"
                        value = line.replace(key, "")
                        new_block[key] = value
                    elif "Results Directory" in line:
                        key = "Results Directory"
                        value = line.replace(key, "")
                        new_block[key] = value
                else:
                    line = line.split("=")
                    if len(line) != 2:
                        print(line)
                    if "variable temperature" in line[1].lower():
                        if "conductivity" in line[0].lower():
                            key = "Heat Conductivity"
                        elif "capacity" in line[0].lower():
                            key = "Heat Capacity"
                        elif "density" in line[0].lower():
                            key = "Density"

                        tabular_data = []

                        continue

                    else:
                        key = line[0].lstrip()
                        value = line[1]
                        new_block[key] = value

        return sif_data

    def map_plasma_heat_flux(
        self, heat_flux_field: HeatFluxField, mapping_tolerance: float | None = None
    ) -> None:
        print("Mapping plasma heat flux to Elmer mesh ...")

        ##########################################################################
        # Prepare IDs for target groups, each plasma triangle will get its own ID
        ##########################################################################
        # get target group_2d IDs
        target_boundaries_group_IDs = []
        for key in self.boundary_conditions:
            target_boundaries_group_IDs.append(int(self.boundary_conditions[key]["group_2d_id"]))

        # get target group_3d IDs
        target_bodies_group_IDs = []
        for key in self.material_properties:
            target_bodies_group_IDs.append(int(self.material_properties[key]["group_3d_id"]))

        # get all boundary groups (not all 2D groups are always included in the FEM case)
        all_boundaries_group_2d_IDs = []
        for b_groups_id in self.mesh['groups_2d'].keys():
            all_boundaries_group_2d_IDs.append(int(b_groups_id.split("_")[-1]))

        # In ElmerFEM, boundary conditions are numbered together with bodies i.e.
        # target body 1, target body 2, target boundary 3 (first target boundary)...

        ##########################################################################
        # Prepare IDs for numbering of BCs in sif file
        ##########################################################################
        bc_ID_plasma = None
        bc_IDs_other = []
        for bc_key in self.boundary_conditions.keys():
            bc_ID = int(bc_key.split("_")[-1])
            #print("BC ID:", bc_ID, "key", bc_key)
            #print(int(self.boundary_conditions[bc_key]["group_2d_id"]), "plasma BC ID", self.plasma_bc_id)
            # skip plasma BC, will be replaced by value to each triangle
            if int(self.boundary_conditions[bc_key]["group_2d_id"]) != self.plasma_bc_id:
                bc_IDs_other.append(bc_ID)
            else:
                bc_ID_plasma = bc_ID

        # Get plasma cells
        plasma_cells_IDs = self.mesh["groups_2d"]["group_2d_id_"+str(self.plasma_bc_id)]
        plasma_cells = self.mesh["cells_2d"][plasma_cells_IDs]
        self.plasma_triangles = plasma_cells
        plasma_mesh = SurfaceMesh(nodes=self.mesh["nodes"], triangles=plasma_cells)
        distance = check_overlap(heat_flux_field.mesh, plasma_mesh, mapping_tolerance)
        print(f"Plasma boundary lies within {distance:.3g} m of the heat flux surface.")

        # Map heat flux to each plasma triangle (nearest field line tracing triangle)
        interp = NearestNDInterpolator(
            heat_flux_field.mesh.cell_centers(), heat_flux_field.heat_flux
        )
        heat_flux = interp(plasma_mesh.cell_centers())

        polydata = to_polydata(plasma_mesh)
        polydata.cell_data["heat_flux"] = heat_flux
        polydata.save(self.case_dir / "plasma_heat_flux.vtp")
        bc_ID_plasma_counter = bc_ID_plasma
        target_boundary_plasma_counter = self.plasma_bc_id

        # Fix first plasma boundary condition
        bc_plasma = self.boundary_conditions["bc_id_"+str(bc_ID_plasma)]
        bc_plasma['group_2d_id'] = self.plasma_bc_id
        bc_plasma['type'] = "heat_flux"
        bc_plasma['value'] = heat_flux[0]
        bc_plasma['value_type'] = "constant"
        
        # Fix first plasma boundary group
        self.mesh["groups_2d"]['group_2d_id_'+str(self.plasma_bc_id)] = plasma_cells_IDs[0]
        self.elmer_mesh_boundary[plasma_cells_IDs[0], 1] = self.plasma_bc_id
        
        # Update counter
        bc_ID_plasma_counter = len(bc_IDs_other+[bc_ID_plasma])
        target_boundary_plasma_counter = len(all_boundaries_group_2d_IDs)

        # update sif and boundary.mesh
        heat_flux_skip_first = heat_flux[1:]
        plasma_cells_IDs_skip_first = plasma_cells_IDs[1:]

        # Update boundary.mesh with new plasma boundary group IDs
        
        target_boundary_plasma_counter_bm = copy.deepcopy(target_boundary_plasma_counter)
        #    self.elmer_mesh_boundary[plasma_cell_ID, 1] = target_boundary_plasma_counter_bm
        #    target_boundary_plasma_counter_bm += 1
    
        for tria_iter in enumerate(self.elmer_mesh_boundary):
            tria = tria_iter[1]
            tria_id = tria_iter[0]
            if tria[1] == self.plasma_bc_id:
                self.elmer_mesh_boundary[tria_id, 1] = target_boundary_plasma_counter_bm
                target_boundary_plasma_counter_bm += 1

        # Update sif file with new plasma boundary conditions and new plasma boundary group IDs
        for tria_ID in range(len(heat_flux_skip_first)):
            # Fix first plasma boundary condition
            self.boundary_conditions["bc_id_"+str(bc_ID_plasma_counter)] = {}
            new_BC = self.boundary_conditions["bc_id_"+str(bc_ID_plasma_counter)]
            new_BC['group_2d_id'] = target_boundary_plasma_counter
            new_BC["type"] = "heat_flux"
            #new_BC["value"] = heat_flux_skip_first[tria_ID]
            new_BC["value"] = heat_flux[tria_ID]
            new_BC['value_type'] = "constant"
            
            # Fix first plasma boundary group
            pos_b_target = [plasma_cells_IDs_skip_first[tria_ID]]
            self.mesh["groups_2d"]['group_2d_id_'+str(target_boundary_plasma_counter)] = np.array(pos_b_target)
            #self.elmer_mesh_boundary[pos_b_target, 1] = target_boundary_plasma_counter

            # Update counter
            bc_ID_plasma_counter +=1
            target_boundary_plasma_counter += 1


        


    def write_sif(self, dir_path: str | Path) -> None:
        file_to_write = Path(dir_path, "case.sif")
        header_block = self.sif_data["Header"]
        simulation_block = self.sif_data["Simulation"]
        constants_block = self.sif_data["Constants"]

        solver_blocks = {}
        for sl in self.sif_data.keys():
            if "Solver" in sl:
                solver_blocks[sl] = self.sif_data[sl]

        equation_blocks = {}
        for eq in self.sif_data.keys():
            if "Equation" in eq:
                equation_blocks[eq] = self.sif_data[eq]

        material_blocks = {}
        for mat in self.sif_data.keys():
            if "Material" in mat:
                material_blocks[mat] = self.sif_data[mat]

        body_blocks = {}
        for bd in self.sif_data.keys():
            if "Body" in bd:
                body_blocks[bd] = self.sif_data[bd]

        with open(file_to_write, 'w') as f:
            f.write("Header\n")
            for key in header_block:
                if "Mesh DB" in key:
                    # Relative to the step directory, where ElmerSolver runs: Elmer's output
                    # writer fails on long absolute paths
                    f.write('Mesh DB "." "elmer_mesh"\n')
                else:
                    f.write("  "+key+" "+header_block[key]+"\n")

            f.write("End\n")
            f.write("\n")
            f.write("Simulation\n")
            for key in simulation_block:
                if self.is_transient and "Simulation Type" in key:
                    f.write("  "+key+" = Transient\n")
                elif self.is_transient and "Timestep intervals" in key:
                    f.write("  "+key+" = "+str(int(self.end_time/self.time_step))+"\n")
                elif self.is_transient and "Timestep Sizes" in key:
                    f.write("  "+key+" = "+str(self.time_step)+"\n")
                else:
                    f.write("  "+key+" = "+simulation_block[key]+"\n")

            f.write("End\n")
            f.write("\n")
            f.write("Constants\n")
            for key in constants_block:
                f.write("  "+key+" = "+constants_block[key]+"\n")

            f.write("End\n")
            f.write("\n")
            for sl in solver_blocks.keys():
                f.write("\n")
                f.write(str(sl)+"\n")
                for key in solver_blocks[sl]:
                    f.write("  "+key+" = "+solver_blocks[sl][key]+"\n")

                f.write("End\n")
                f.write("\n")
            # Write another solver that will store temperature results
            # after each time step
            f.write('Solver '+str(len(solver_blocks)+1)+'\n')
            f.write(' Exec Solver = String "after timestep"\n')
            f.write(' exec interval = 1\n')
            f.write(' Equation = String "ResultOutput"\n')
            f.write(' Procedure = File "ResultOutputSolve" "ResultOutputSolver"\n')
            f.write(' Output File Name = File "elmer_temperatures."\n')
            f.write(' Output Format = String "vtk"\n')
            f.write(' Vtk Format = Logical True\n')
            f.write('End\n')

            
            for eq in equation_blocks.keys():
                f.write("\n")
                f.write(str(eq)+"\n")
                for key in equation_blocks[eq]:
                    if "Active Solvers" in key:
                        f.write("  Active Solvers(2) = 1 2\n")
                    else:
                        f.write("  "+key+" = "+equation_blocks[eq][key]+"\n")

                f.write("End\n")
                
            for bd in body_blocks.keys():
                f.write("\n")
                f.write(str(bd)+"\n")
                for key in body_blocks[bd]:
                    f.write("  "+key+" = "+body_blocks[bd][key]+"\n")

                f.write("End\n")

            for mat in material_blocks.keys():
                f.write("\n")
                f.write(str(mat)+"\n")
                for key in material_blocks[mat]:
                    if type(material_blocks[mat][key]) == np.ndarray:
                        # write tabular data
                        f.write("  "+key+" = Variable Temperature\n")
                        f.write("  Real\n")

                        for row in material_blocks[mat][key]:
                            f.write("    "+str(row[0])+" "+str(row[1])+"\n")

                        f.write("  End\n")
                    else:
                        f.write("  "+key+" = "+material_blocks[mat][key]+"\n")

                    
                f.write("End\n")
            # Now rewrite boundary conditions with updated plasma heat
            # flux values
            #self.boundary_conditions["bc_id_"+str(bc_ID_plasma_counter)] = {}
            #new_BC = self.boundary_conditions["bc_id_"+str(bc_ID_plasma_counter)]
            #new_BC['group_2d_id'] = target_boundary_plasma_counter
            #new_BC["type"] = "heat_flux"
            #new_BC["value"] = heat_flux_skip_first[tria_ID]
            #new_BC['value_type'] = "constant"
            for bc_key in self.boundary_conditions.keys():
                f.write("\n")
                # add 1 to convert back to 1-based indexing for sif file
                f.write("Boundary Condition "+str(int(bc_key.split("_")[-1])+1)+"\n") 
                target_boundary = self.boundary_conditions[bc_key]["group_2d_id"]+1
                
                f.write("  Target boundaries(1) = "+str(target_boundary)+"\n")
                f.write('  Name = "boundary'+str(target_boundary)+'"\n')

                bc_type = self.boundary_conditions[bc_key]["type"]
                if bc_type.lower() == "heat_flux":
                    if self.is_transient:
                        f.write("  Heat Flux = Variable Time\n")
                        f.write("  Real\n")
                        time = 0
                        while time <= self.end_time:
                            f.write("    "+str(time)+" "+str(self.boundary_conditions[bc_key]["value"])+"\n")
                            time += self.time_step
                        f.write("  End\n")
                    else:
                        f.write("  Heat flux = "+str(self.boundary_conditions[bc_key]["value"])+"\n")
                elif bc_type.lower() == "htc":
                    f.write("  Heat transfer coefficient = "+str(self.boundary_conditions[bc_key]["value"])+"\n")
                    f.write("  External temperature = "+str(self.boundary_conditions[bc_key]["external_value"])+"\n")
                elif bc_type.lower() == "temperature":
                    f.write("  Temperature = "+str(self.boundary_conditions[bc_key]["value"])+"\n")
                f.write("End\n")


    def write_elmer_mesh(self, dir_path: str | Path) -> None:
        dir_path = Path(dir_path) / Path("elmer_mesh")
        dir_path.mkdir(parents=True, exist_ok=True)
        # In "mesh.nodes" file, node numbering starts from 1, so add 1
        # to all node IDs. Frst and second columns are of type 'int'
        node_IDs = self.elmer_mesh_nodes[:, 0].astype(int) + 1
        node_connectivity = self.elmer_mesh_nodes[:, 1].astype(int)
        node_x = self.elmer_mesh_nodes[:, 2]
        node_y = self.elmer_mesh_nodes[:, 3]
        node_z = self.elmer_mesh_nodes[:, 4]
        with open(os.path.join(dir_path, "mesh.nodes"), 'w') as f:
            for i in range(len(self.elmer_mesh_nodes)):
                f.write(f"{node_IDs[i]} {node_connectivity[i]} {node_x[i]} {node_y[i]} {node_z[i]}\n")
                
        #for i in range(len(self.elmer_mesh_nodes)):   
        #np.savetxt(os.path.join(dir_path, "mesh.nodes"), self.elmer_mesh_nodes, fmt="%f")

        np.savetxt(os.path.join(dir_path, "mesh.elements"), self.elmer_mesh_elements+1, fmt="%d")
        np.savetxt(os.path.join(dir_path, "mesh.boundary"), self.elmer_mesh_boundary+1, fmt="%d")

    def run(
        self,
        dir_path: str | Path,
        heat_flux: HeatFluxField,
        mapping_tolerance: float | None = None,
    ) -> None:
        """Read sif and mesh, map plasma heat flux to BCs, write the case and run ElmerSolver."""
        self.case_dir = Path(dir_path)
        self.sif_data = self.read_sif(self.sif_file)
        self.get_elmer_mesh(self.mesh_type, self.mesh_file, self.case_dir)
        self.mesh = self.get_tm_mesh()
        self.material_properties = get_material_properties(self.sif_data)
        self.boundary_conditions = get_boundary_conditions(self.sif_data)

        self.map_plasma_heat_flux(heat_flux, mapping_tolerance)
        self.write_sif(dir_path)

        self.write_elmer_mesh(dir_path)

        print("Running ElmerSolver ...")
        result = subprocess.run( [elmer_executable("ElmerSolver"), "case.sif"],
                                 cwd=dir_path,
                                 capture_output=True,
                                 text=True)

        print(result.stdout)
        print(result.stderr)
        if result.returncode != 0:
            raise RuntimeError("ElmerSolver failed with return code {}".format(result.returncode))

        self.result = self.read_temperature()

    def read_temperature(self) -> TemperatureField:
        """Read nodal temperature on the plasma boundary from the ResultOutputSolver files.

        ElmerSolver writes ``elmer_mesh/elmer_temperatures.NNNN.vtk`` (one file per time step,
        points in the order of ``mesh.nodes``); only the plasma triangles are kept.
        """
        if self.case_dir is None or self.plasma_triangles is None:
            raise RuntimeError("ElmerFEM has not been run yet.")

        output_dir = self.case_dir / "elmer_mesh"
        files = sorted(
            output_dir.glob("elmer_temperatures.*.vtk"), key=lambda p: int(p.suffixes[-2][1:])
        )
        if not files:
            raise FileNotFoundError(f"No 'elmer_temperatures.*.vtk' files in {output_dir}")

        # Keep only the nodes of the plasma triangles and renumber them from 0
        used_nodes = np.unique(self.plasma_triangles)
        new_index = np.full(len(self.mesh["nodes"]), -1)
        new_index[used_nodes] = np.arange(len(used_nodes))
        mesh = SurfaceMesh(
            nodes=self.mesh["nodes"][used_nodes], triangles=new_index[self.plasma_triangles]
        )

        temperature = np.empty((len(files), len(used_nodes)))
        for i, file in enumerate(files):
            output = pv.read(file)
            if output.n_points != len(self.mesh["nodes"]):
                raise ValueError(
                    f"{file} has {output.n_points} points, mesh has {len(self.mesh['nodes'])}"
                )
            temperature[i] = np.asarray(output.point_data["Temperature"])[used_nodes]

        if self.is_transient:
            times = self.time_step * np.arange(1, len(files) + 1)  # written after each step
        else:
            times = np.zeros(len(files))
        return TemperatureField(mesh=mesh, times=times, temperature=temperature)

    def get_result(self) -> TemperatureField:
        """Plasma-facing surface temperature: one row per written time step."""
        if self.result is None:
            raise RuntimeError("ElmerFEM has not been run yet.")
        return self.result
