from BackendElmerFEM import BackendElmerFEM

mesh_file = "/home/brankm/work/heatfem/kvader_tungsten_steel/elmer_kvader_tungsten_steel_dense"
mesh_type = "mesh"
sif_file = "/home/brankm/work/heatfem/kvader_tungsten_steel/case_denser.sif"
plasma_bc_id = 3 # 2d group of plasma facing surface

b_ef = BackendElmerFEM(mesh_file, mesh_type, sif_file, plasma_bc_id)

b_ef.map_plasma_heat_flux([])
