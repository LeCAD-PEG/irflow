import numpy as np
import os
import re
import riptrace_py
from riptrace_py.target import Target
from riptrace_py.shadow import Shadow
from riptrace_py.utils import read_dat_file
from pathlib import Path
import subprocess

class BackendRiptrace(object):
    """Documentation for BackendRiptrace
2
    """
    def __init__(self, target_mesh_path, shadow_mesh_path,
                 equilibrium_path,
                 target_mesh_type,
                 shadow_mesh_type, equilibrium_type,
                 power_loss, lambda_main,
                 point_omp):
        super(BackendRiptrace, self).__init__()

        self.target_mesh_path = target_mesh_path
        self.target_mesh_type = target_mesh_type
        self.shadow_mesh_path = shadow_mesh_path
        self.shadow_mesh_type = shadow_mesh_type
        self.equilibrium_path = equilibrium_path
        self.equilibrium_type = equilibrium_type

        self.power_loss = power_loss # Power in the scrape-off layer [W]
        self.lambda_main = lambda_main # main decay length [m]
        self.point_omp = point_omp # Location of separatrix/LCFS at outer midplane R, Z[m]

        self.target_mesh = None
        self.shadow_mesh = None
        self.numerical_params = None

    def get_flt_target_mesh(self, ):
        target = Target(name="irflow_target")
        if self.target_mesh_type.lower == "dat":
            target.load_dat(self.target_mesh_path)
        elif self.target_mesh_type.lower == "vtk":
            target.load_vtk(self.target_mesh_path)

        self.target_mesh = {"nodes": target.nodes, "cells_2d": target.triangles}
        return self.target_mesh

    def set_flt_target_mesh(self, ):
        pass

    def get_flt_shadow_mesh(self, ):
        shadow = Shadow(name="irflow_shadow")
        if self.shadow_mesh_type.lower == "dat":
            shadow.load_dat(self.shadow_mesh_path)
        elif self.shadow_mesh_type.lower == "vtk":
            shadow.load_vtk(self.shadow_mesh_path)

        self.shadow_mesh = {"nodes": shadow.nodes, "cells_2d": shadow.triangles}
        return self.shadow_mesh

    def set_flt_shadow_mesh(self, ):
        pass

    def get_equilibrium(self, ):
        if self.equilibrium_type.lower() == "pfcflux":
            from riptrace_py.io_utils import read_pfcflux_data
            magnetic_field, magnetic_flux = read_pfcflux_data(self.equilibrium_path)

        elif self.equilibrium_type.lower() == "magtfm":
            from riptrace_py.mag_io_reader import read_magtfm_data
            fiesta_file_path = self.equilibrium_path[0]
            ov5_file_path = self.equilibrium_path[1]
            magnetic_field, magnetic_flux = read_magtfm_data(fiesta_file_path,
                                                             ov5_file_path)

        #elif self.equilibrium_type.lower() == "energopul":
        #    pass

        else:
            [magnetic_field, magnetic_flux] = self.equilibrium_path

        equilibrium_dict = {"phi_1d": magnetic_field.phi_1d,
                            "z_1d": magnetic_field.z_1d,
                            "r_1d": magnetic_field.r_1d,
                            "bphi_1d": magnetic_field.bphi_1d,
                            "bz_1d": magnetic_field.bz_1d,
                            "br_1d": magnetic_field.br_1d,
                            "psi_1d": magnetic_flux.psi_1d}
        
        return equilibrium_dict

    def set_equilibrium(self):
        pass

    def get_numerical_params(self):
        if self.numerical_params is None:
            from mag_riptrace_case import RipTraceNumericalParams
            self.numerical_params = RipTraceNumericalParams()

        data = {"h_initial": self.numerical_params.h_initial,
                "tolerance": self.numerical_params.tolerance,
                "h_min": self.numerical_params.h_min,
                "h_max": self.numerical_params.h_max,
                "max_points": self.numerical_params.max_points,
                "max_length": self.numerical_params.max_length,
                "is_shadowed": self.numerical_params.is_shadowed,
                "offset": self.numerical_params.offset,
                "kdtree_max_depth": self.numerical_params.kdtree_max_depth,
                "debug_vtk":  self.numerical_params.debug_vtk}

        return data

    def set_numerical_params(self):
        pass

    def get_irflow_data(self, ):
        target = self.get_flt_target_mesh()
        shadow = self.get_flt_shadow_mesh()

        equilibrium = self.get_equilibrium()
        equilibrium["power_loss"] = self.power_loss
        equilibrium["lambda_main"] = self.lambda_main
        equilibrium["point_omp"] = self.point_omp
        numerical_parameters = self.get_numerical_params()

        flt_data = {"target": target,
                    "shadow": shadow,
                    "equilibrium": equilibrium,
                    "num_params": numerical_parameters}

        return flt_data

    def prepare(self, case_dir):
        path = Path(case_dir)
        if not path.is_dir():
            path.mkdir(parents=True)

        # Make process and execute riptrace in case_dir
        with subprocess.Popen(
                ["riptrace"],
                cwd=case_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
        ) as proc:
            for line in proc.stdout:
                print(line, end="")
