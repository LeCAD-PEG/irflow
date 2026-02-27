import numpy as np
import os
import re


# Connector between IR workflow - thermal model and Elmer
class BackendElmerFEM(object):
    """Documentation for BackendElmerFEM

    """
    def __init__(self, mesh_file, sif_file, mesh_type):
        super(BackendElmerFEM, self).__init__()
        self.mesh_file = mesh_file
        self.sif_file = sif_file
        self.mesh_type = mesh_type

        self.sif_data = None
        
    def get_tm_mesh(self):
        if self.mesh_type.lower() == "elmerfem":
            nodes = np.loadtxt(os.path.join(self.mesh_file, "mesh.nodes"))[-3:, :]

            elements = np.loadtxt(os.path.join(self.mesh_file, "mesh.elements"))
            cells_3d = elements[:, -4:]
            groups_3d = {}
            for element in enumerate(elements[:, 1]):
                iterator = element[0]
                element_id = element[1]
                if element_id in groups_3d.keys():
                    groups_3d["group_3d_id_"+str(element_id)] = [iterator]
                else:
                    groups_3d["group_3d_id_"+str(element_id)].append(iterator)

            for key in groups_3d.keys():
                groups_3d[key] = np.array(groups_3d[key])

            boundaries = np.loadtxt(os.path.join(self.mesh_file, "mesh.boundary"))
            cells_2d = boundaries[:, -3:]
            groups_2d = {}
            for boundary in enumerate(boundaries[:, 1]):
                iterator = boundary[0]
                boundary_id = boundary[1]
                if boundary_id in groups_3d.keys():
                    groups_2d["group_2d_id_"+str(boundary_id)] = [boundaries[iterator, -4:]]
                else:
                    groups_2d["group_2d_id_"+str(boundary_id)].append(boundaries[iterator, -4:])

            for key in groups_2d.keys():
                groups_2d[key] = np.array(groups_2d[key])

            self.mesh = {"nodes": nodes,
                         "cells_3d": cells_3d,
                         "cells_2d": cells_2d,
                         "groups_3d": groups_3d,
                         "groups_2d": groups_2d}

            return self.mesh


    def read_sif(self, filename=None):
        if not filename:
            filename = self.sif_file
            

        with open(filename, "r") as f:
            f_content = f.readlines().split("\n")
            
        self.sif_data = data
    
