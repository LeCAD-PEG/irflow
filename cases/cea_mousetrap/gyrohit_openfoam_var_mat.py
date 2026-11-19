"""CEA mousetrap: GYROHIT heat flux, transient OpenFOAM thermal model, per-region materials.

The OpenFOAM bashrc is taken from the code-paths file (see templates/code_paths.toml).
"""

import numpy as np

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

# Temperature [K], heat capacity [J/(kg K)]
heat_capacity_table = np.array(
    [
        [297, 666],
        [373, 812],
        [473, 937],
        [573, 1022],
        [673, 1084],
        [773, 1131],
        [873, 1167],
        [973, 1198],
        [1073, 1223],
        [1173, 1245],
        [1273, 1263],
        [1373, 1279],
        [1473, 1294],
        [2000, 1373],
        [3000, 1523],
        [4000, 1673],
        [5000, 1823],
        [6000, 1973],
        [7000, 2073],
        [8000, 2173],
    ]
)

# k(T) = sum(c_n * T^n) [W/(m K)]
heat_conductivity_polynomial = np.array(
    [2.177878e01, -1.422731e-02, 8.277203e-06, -2.795318e-09, 0, 0, 0, 0]
)

material_property_SiO2_variable = {
    "density": {"type": "constant", "value": 3060},
    "heat_capacity": {"type": "table", "value": heat_capacity_table},
    "heat_conductivity": {"type": "polynomial", "value": heat_conductivity_polynomial},
}

# Constant properties are used for now; swap in material_property_SiO2_variable to test k(T)
material_property_SiO2 = {
    "density": {"type": "constant", "value": 3060},
    "heat_capacity": {"type": "constant", "value": 1000},
    "heat_conductivity": {"type": "constant", "value": 5},
}

regions = [
    "housing_2",
    "sample",
    "pions",
    "housing_1",
    "screws",
    "region1",
    "region2",
    "region3",
    "region4",
]

config_tm = {
    "mesh_file": DATA_DIR + "Mesh_groups_translated.unv",
    "mesh_type": "unv",
    "plasma_bc_name": "plasmaBC",
    "time_stepping": {"type": "transient", "time_step": 0.001, "end_time": 0.2},
    "initial_temperature": 373.0,
    "boundary_conditions": {
        "plasmaBC": {"type": "heat_flux", "value": None},
        "adiabaticBC": {"type": "heat_flux", "value": 0.001, "value_type": "constant"},
        "const_T": {"type": "temperature", "value": 373, "value_type": "constant"},
    },
    "material_properties": {region: material_property_SiO2 for region in regions},
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
