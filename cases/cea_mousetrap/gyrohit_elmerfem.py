"""CEA mousetrap: GYROHIT heat flux followed by a transient ElmerFEM thermal model."""

from irflow import FieldLineTracing, IRFlow, ThermalModel

DATA_DIR = "/home/brankm/work/CEA_mousetrap/"

config = {
    "target_mesh_path": DATA_DIR + "target.vtk",
    "target_mesh_type": "vtk",
    "shadow_mesh_path": None,
    "shadow_mesh_type": None,
    "equilibrium_path": None,
    "equilibrium_type": None,
}

flt_case = FieldLineTracing(
    name="gyrohit_cea_mousetrap", enabled=True, backend_code="gyrohit", config=config
)

config_tm = {
    "mesh_file": DATA_DIR + "Mesh_37229tetra.unv",
    "mesh_type": "unv",
    "sif_file": DATA_DIR + "case_const_mat_prop.sif",
    "plasma_bc_id": 2,
    "time_stepping": {"type": "transient", "time_step": 0.1, "end_time": 0.5},
}

tm_case = ThermalModel("mastu_elmerfem", enabled=True, backend_code="elmerfem", config=config_tm)

irflow_case = IRFlow(
    case_name="CEA_mousetrap",
    root_dir="/home/brankm/data/irflow_studies/",
    steps=[flt_case, tm_case],
    overwrite=True,
)

irflow_case.run()
