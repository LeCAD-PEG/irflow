"""CEA mousetrap: GYROHIT heat flux followed by a transient OpenFOAM thermal model.

The OpenFOAM bashrc is taken from the code-paths file (see templates/code_paths.toml).
"""

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
    "mesh_file": DATA_DIR + "Mesh_groups_38000tetra.unv",
    "mesh_type": "unv",
    "plasma_bc_name": "plasma_BC",
    "time_stepping": {"type": "transient", "time_step": 0.1, "end_time": 0.5},
    "initial_temperature": 300.0,
    "boundary_conditions": {
        "plasma_BC": {"type": "heat_flux", "value": None},
        "const_T": {"type": "htc", "value": 17283, "T_ext": 300.0, "value_type": "constant"},
    },
}

tm_case = ThermalModel(
    "cea_mousetrap_transient", enabled=True, backend_code="openfoam", config=config_tm
)

irflow_case = IRFlow(
    case_name="CEA_mousetrap_transient",
    root_dir="/home/brankm/work/irflow_studies/",
    steps=[flt_case, tm_case],
    overwrite=True,
)

irflow_case.run()
