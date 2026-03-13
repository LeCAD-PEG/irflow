from flow import IRFlow
from field_line_tracing import FieldLineTracing

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

irflow_case = IRFlow(case_name="test_workflow_flt_only",
                     root_dir="/home/brankm/work/irflow_studies/",
                     steps=[flt_case])

irflow_case.run()
