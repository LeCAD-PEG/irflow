import numpy as np
import os
import re
import riptrace_py
from riptrace_py.mag_riptrace_mesh import Target
from riptrace_py.mag_riptrace_mesh import Shadow
#from riptrace_py.utils import read_dat_file
from riptrace_py.mag_riptrace_case import RipTraceCase
from pathlib import Path
import subprocess


class BackendRiptrace(object):
    """Documentation for BackendRiptrace

    """
    def __init__(self, target_mesh_path, shadow_mesh_path,
                 equilibrium_path,
                 target_mesh_type,
                 shadow_mesh_type, equilibrium_type,
                 power_loss, lambda_main,
                 point_omp, numerical_parameters=None):
        super(BackendRiptrace, self).__init__()

        # input parameters
        self.target_mesh_path = target_mesh_path
        self.target_mesh_type = target_mesh_type
        self.shadow_mesh_path = shadow_mesh_path
        self.shadow_mesh_type = shadow_mesh_type
        self.equilibrium_path = equilibrium_path
        self.equilibrium_type = equilibrium_type

        self.power_loss = power_loss # Power in the scrape-off layer [W]
        self.lambda_main = lambda_main # main decay length [m]
        self.point_omp = point_omp # Location of separatrix/LCFS at outer midplane R, Z[m]
        self.numerical_params = numerical_parameters # numerical parameters for FLT

        # backend data format (private)
        self.__target_riptrace = None
        self.__shadow_riptrace = None
        self.__magnetic_field_riptrace = None
        self.__magnetic_flux_riptrace = None
        self.__case_riptrace = None

        # irflow data format (to be passed back to FLT class)
        self.target_irflow = None
        self.shadow_irflow = None
        self.equilibrium_irflow = None

    def get_target_mesh_riptrace(self):
        self.__target_riptrace = Target(name="irflow_target")
        if self.target_mesh_type.lower() == "dat":
            self.__target_riptrace.load_dat(self.target_mesh_path)
        elif self.target_mesh_type.lower() == "vtk":
            self.__target_riptrace.load_vtk_mesh(self.target_mesh_path)
        else:
            # Throw exception
            raise ValueError( f"Unknown target mesh type '{self.target_mesh_type}'. "
                              "Supported types: 'vtk' and 'dat'." )


    def get_target_mesh_irflow(self, ):
        if not self.__target_riptrace:
            self.get_target_mesh_riptrace()

        self.target_irflow = {"nodes": self.__target_riptrace.nodes,
                              "cells_2d": self.__target_riptrace.triangles}
        return self.target_irflow

    def set_flt_target_mesh(self, ):
        pass

    def get_shadow_mesh_riptrace(self, ):
        self.__shadow_riptrace = Shadow(name="irflow_shadow")
        if self.shadow_mesh_type.lower() == "dat":
            self.__shadow_riptrace.load_dat(self.shadow_mesh_path)
        elif self.shadow_mesh_type.lower() == "vtk":
            self.__shadow_riptrace.load_vtk_mesh(self.shadow_mesh_path)

        else:
            # Throw exception
            raise ValueError( f"Unknown shadow mesh type '{self.shadow_mesh_type}'. "
                              "Supported types: 'vtk' and 'dat'." )

    def get_shadow_mesh_irflow(self, ):
        if not self.__shadow_riptrace:
            self.get_shadow_mesh_riptrace()

        self.shadow_irflow = {"nodes": self.__shadow_riptrace.nodes,
                              "cells_2d": self.__shadow_riptrace.triangles}
        return self.shadow_irflow

    def set_flt_shadow_mesh(self, ):
        pass

    def get_equilibrium(self, ):
        if self.equilibrium_type.lower() == "pfcflux":
            from riptrace_py.mag_io_reader import read_pfcflux_data
            self.__magnetic_field_riptrace, self.__magnetic_flux_riptrace = \
                read_pfcflux_data(self.equilibrium_path.get("txt"))

        elif self.equilibrium_type.lower() == "smardda":
            from riptrace_py.mag_io_reader import read_magtfm_data
            fiesta_file_path = self.equilibrium_path.get("fiesta")
            ov5_file_path = self.equilibrium_path.get("ov5")
            self.__magnetic_field_riptrace, self.__magnetic_flux_riptrace =\
                read_magtfm_data(fiesta_file_path,
                                 ov5_file_path)

        elif self.equilibrium_type.lower() == "energopul":
            # TODO
            raise NotImplementedError( "Equilibrium type 'energopul' is "
                                       "recognized but not implemented yet." )            
        else:
            # Throw exception
            raise ValueError( f"Unknown equilibrium type '{self.equilibrium_type}'. "
                              "Supported types: 'pfcflux' and 'smardda'." )

        equilibrium_dict = {"phi_1d": self.__magnetic_field_riptrace.phi_1d,
                            "z_1d": self.__magnetic_field_riptrace.z_1d,
                            "r_1d": self.__magnetic_field_riptrace.r_1d,
                            "bphi_1d": self.__magnetic_field_riptrace.bphi_1d,
                            "bz_1d": self.__magnetic_field_riptrace.bz_1d,
                            "br_1d": self.__magnetic_field_riptrace.br_1d,
                            "psi_1d": self.__magnetic_flux_riptrace.psi_1d}
        
        return equilibrium_dict

    def set_equilibrium(self):
        pass

    def get_numerical_params(self):
        if self.numerical_params is None:
            from riptrace_py.mag_riptrace_case import RipTraceNumericalParams
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
        target = self.get_target_mesh_irflow()
        shadow = self.get_shadow_mesh_irflow()

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

    def run(self, case_dir):
        #path = Path(case_dir)
        #if not path.is_dir():
        #    path.mkdir(parents=True)

        # Get proper data in riptrace format from input files
        # Verify data?
        self.get_target_mesh_riptrace()
        self.get_shadow_mesh_riptrace()
        self.get_equilibrium()

        # Run riptrace inside python
        riptrace_case_dir = Path(case_dir) / "riptrace_case"
        self.case_riptrace = RipTraceCase("smardda_case", self.__target_riptrace,
                                          self.__shadow_riptrace,
                                          self.__magnetic_field_riptrace,
                                          self.__magnetic_flux_riptrace, self.power_loss,
                                          self.lambda_main, None, riptrace_case_dir)

        # Compute wetted area
        self.case_riptrace.compute_wetted_area()

        # Compute power deposition
        self.case_riptrace.compute_power_deposition(self.power_loss,
                                                    self.lambda_main, 1/2,
                                                    self.point_omp)

    def get_heat_flux(self):
        return self.case_riptrace.target.Q
        
