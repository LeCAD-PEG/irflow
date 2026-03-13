import pdb
from io_utils.io_utils import read_vtk

np = __import__("numpy")

class BackendGyrohit:
    def __init__(self, target_mesh_path, shadow_mesh_path,
                 equilibrium_path,
                 target_mesh_type,
                 shadow_mesh_type, equilibrium_type):
        self.target_mesh_path = target_mesh_path
        self.target_mesh_type = target_mesh_type
        self.shadow_mesh_path = shadow_mesh_path
        self.shadow_mesh_type = shadow_mesh_type
        self.equilibrium_path = equilibrium_path
        self.equilibrium_type = equilibrium_type

        # irflow data format (to be passed back to FLT class)
        self.target_irflow = self.get_target_mesh_irflow()

    def get_target_mesh_gyrohit(self):
        if self.target_mesh_type.lower() == "vtk":
            # Load VTK mesh using appropriate library (e.g., PyVista, VTK)
            pass

    def get_target_mesh_irflow(self):
        if self.target_mesh_type.lower() == "vtk":
            # Load VTK mesh using appropriate library (e.g., PyVista, VTK)
            data = read_vtk(self.target_mesh_path)
            return {"nodes": data["nodes"],
                    "cells_2d": data["cells"]}

    def run(self, dir_path):
        if self.shadow_mesh_path == None and self.equilibrium_path == None:
            print("Gyrohit: Shadow and equilibrium not " \
                  "defined. Processing heat fluxes in target mesh.")
            data = read_vtk(self.target_mesh_path)

            self.target_irflow = {"nodes": data["nodes"],
                                  "cells_2d": data["cells"],
                                  "heat_flux": data["Heat-flux[W/m^2]_cells"]}

            #self.write_of_heat_flux()

    def get_irflow_data(self, ):
                    
        data = {"target":{"nodes": self.target_irflow["nodes"],
                          "cells_2d": self.target_irflow["cells_2d"]}}
        #data["heat_flux"] = self.get_heat_flux()
        return data

    def get_heat_flux(self):
        return self.target_irflow["heat_flux"]

    def write_of_heat_flux(self):
        with open("/home/brankm/work/kokovnik_clanek/OpenFOAM/CEA_mousetrap/1_8/constant/data.txt", "w") as f:
            f.write("time: 0.0\n")

            nodes = np.array(self.target_irflow["nodes"])      # shape: (n_nodes, 3)
            cells = self.target_irflow["cells_2d"]             # list of lists of node indices
            
            cell_centers = []
            
            for cell_iter in enumerate(cells):
                cell_id = cell_iter[0]
                cell = cell_iter[1]
                cell = [int(i) for i in cell]
                pts = nodes[cell]              # izbereš koordinate vseh vozlišč celice
                centroid = pts.mean(axis=0)    # povprečje po vrsticah
                cell_centers.append(centroid)
                f.write(str(centroid[0])+" "+str(centroid[1])+" "+str(centroid[2])+" " \
                        +str(self.target_irflow["heat_flux"][cells.index(cell)])+"\n")
