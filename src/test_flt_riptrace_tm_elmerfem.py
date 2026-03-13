from BackendRiptrace import BackendRiptrace
from BackendElmerFEM import BackendElmerFEM
from flow import IRFlow
from field_line_tracing import FieldLineTracing
from thermal_model import ThermalModel


###############################################################################
#                                Test RipTrace                                #
###############################################################################

config = {"target_mesh_path": "/home/brankm/work/l2g/benchmarks/"
                              "Test_MNEW8_sha12_res12/target_res12_m.dat",
          "target_mesh_type": "dat",
          "shadow_mesh_path": "/home/brankm/work/l2g/benchmarks/"
                              "Test_MNEW8_sha12_res12/sha12_m_360deg.vtk",
          "shadow_mesh_type": "vtk",
          "equilibrium_path": {"ov5": "/home/brankm/work/l2g/benchmarks/"
                               "Test_MNEW8_sha12_res12/testov5.txt",
                               "fiesta": "/home/brankm/work/l2g/benchmarks/"
                               "Test_MNEW8_sha12_res12/MNEW8.equ"},
          "equilibrium_type": "smardda",
          "power_loss": 1e6,  # Power in the scrape-off layer [W]
          "lambda_main": 0.01,  # main decay length [m]
          "point_omp": [1.3391667, 0]}  # Location of separatrix/LCFS
                                        # at outer midplane R, Z[m]

flt_case = FieldLineTracing(name="riptrace_test_mastu", enabled=True,
                            backend_code="riptrace", config=config,
                            resources=None)



#mesh_file = "/home/brankm/work/irflow/src/examples/mastu_test/Mesh_1_scaled_m.unv"
#mesh_type = "mesh"
#sif_file = "/home/brankm/work/heatfem/kvader_tungsten_steel/case_denser.sif"
#plasma_bc_id = 3 # 2d group of plasma facing surface

config_tm = {"mesh_file": "/home/brankm/work/irflow/src/examples/mastu_test/Mesh_1_scaled_m.unv",
             "mesh_type": "unv",
             "sif_file": "/home/brankm/work/irflow/src/examples/mastu_test/case.sif",
             "plasma_bc_id": 1
             }

tm_case = ThermalModel("mastu_elmerfem", enabled=True,
                       backend_code="elmerfem", config=config_tm,
                       root_dir="/home/brankm/work/irflow_studies/test_workflow_flt_and_tm/ThermalModel")

irflow_case = IRFlow(case_name="test_workflow_flt_and_tm",
                     root_dir="/home/brankm/work/irflow_studies/",
                     steps=[flt_case, tm_case])

irflow_case.run()
