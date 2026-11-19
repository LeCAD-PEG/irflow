"""CEA mousetrap: GYROHIT heat flux followed by the fast simple1d thermal model.

Every triangle of the GYROHIT target mesh is treated as an independent 1D slab (no lateral
heat conduction), so this runs in seconds and needs no external thermal code.
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

material_property_SiO2 = {
    "density": {"type": "constant", "value": 3060},
    "heat_capacity": {"type": "constant", "value": 1000},
    "heat_conductivity": {"type": "constant", "value": 5},
}

config_tm = {
    "thickness": 0.01,
    "material_properties": material_property_SiO2,
    "back_boundary": {"type": "temperature", "value": 373.0},
    "initial_temperature": 373.0,
    "time_stepping": {"type": "transient", "time_step": 0.001, "end_time": 0.2},
    # In 0.2 s the heat penetrates only ~1 mm of the 10 mm slab: a fine cell at the heated
    # surface that grows towards the back resolves this with few points
    "n_points": 51,
    "first_cell_size": 5e-6,
}

tm_case = ThermalModel(
    "cea_mousetrap_simple1d", enabled=True, backend_code="simple1d", config=config_tm
)

irflow_case = IRFlow(
    case_name="CEA_mousetrap_simple1d",
    root_dir="/home/brankm/work/irflow_studies/",
    steps=[flt_case, tm_case],
    overwrite=True,
)

irflow_case.run()
