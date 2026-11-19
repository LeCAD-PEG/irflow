from base import BackendElmerFEM

mesh_file = "/home/brankm/work/heatfem/kvader_tungsten_steel/elmer_kvader_tungsten_steel_dense"
sif_file = "/home/brankm/work/heatfem/kvader_tungsten_steel/case_denser.sif"
mesh_type = "elmerfem"

elmerfem_case = BackendElmerFEM(mesh_file, sif_file, mesh_type)





