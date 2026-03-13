from flow import IRFlow
from field_line_tracing import FieldLineTracing
from thermal_model import ThermalModel

###############################################################################
#                                Test Gyrohit                                #
###############################################################################

config = {"target_mesh_path": "/home/brankm/work/CEA_mousetrap/target.vtk",
          "target_mesh_type": "vtk",
          "shadow_mesh_path": None,
          "shadow_mesh_type": None,
          "equilibrium_path": None,
          "equilibrium_type": None}

flt_case = FieldLineTracing(name="gyrohit_cea_mousetrap", enabled=True,
                            backend_code="gyrohit", config=config,
                            resources=None)


config_tm = {"mesh_file": "/home/brankm/work/CEA_mousetrap/Mesh_37229tetra.unv",
             "mesh_type": "unv",
             "sif_file": "/home/brankm/work/CEA_mousetrap/case_const_mat_prop.sif",
             "plasma_bc_id": 2,
             "time_stepping": {"type": "transient", "time_step": 0.1, "end_time": 0.5}#{"type": "steady", "time_step": None, "end_time": None},
             }

tm_case = ThermalModel("mastu_elmerfem", enabled=True,
                       backend_code="elmerfem", config=config_tm,
                       root_dir="/home/brankm/work/irflow_studies/CEA_mousetrap/ThermalModel")

irflow_case = IRFlow(case_name="CEA_mousetrap",
                     root_dir="/home/brankm/work/irflow_studies/",
                     steps=[flt_case, tm_case])

irflow_case.run()
