"""CEA mousetrap: GYROHIT heat flux -> OpenFOAM thermal model -> raysect IR camera.

The camera looks from above into the divertor gap where the RFA sits, straight down onto the
top of the RFA. There is no real camera calibration yet, so the camera is an ideal pinhole
with an assumed position; replace it by the calibrated camera when the data is available.

Geometry (from the meshes, in metres): the RFA ("sample" region) is the block
0.015 < x < 0.017, 0.00725 < y < 0.01725, 0 < z < 0.012. It sticks out of the housing (top at
z = 0.005), so the heated plasma-facing surface is its top face (z = 0.012) and the upper 7 mm
of its four sides. Plasma heat arrives from +z.

The housing around the RFA is in the optical scene (it occludes, emits and reflects); build
its surface once with ``make_housing_scene.py`` before running this case.

The result is one camera image per thermal time step (201 images, 0 to 0.2 s) in
``<case>/OpticalModel/images.pvd``, with radiance and apparent temperature. The OpenFOAM bashrc
comes from the code-paths file (see templates/code_paths.toml).
"""

from pathlib import Path

import numpy as np

from irflow import FieldLineTracing, IRFlow, OpticalModel, ThermalModel

CASE_DIR = Path(__file__).resolve().parent
DATA_DIR = "/home/brankm/work/CEA_mousetrap/"

# --- Field line tracing: precomputed GYROHIT heat flux on the RFA surface -------------------
config_flt = {"target_mesh_path": DATA_DIR + "target.vtk", "target_mesh_type": "vtk"}

flt_case = FieldLineTracing(
    name="gyrohit_cea_mousetrap", enabled=True, backend_code="gyrohit", config=config_flt
)

# --- Thermal model: OpenFOAM, transient, all regions SiO2-like with constant properties ------
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
    "cea_mousetrap_openfoam", enabled=True, backend_code="openfoam", config=config_tm
)

# --- Optical model: IR camera above the divertor gap, looking down onto the RFA top ---------
RFA_TOP_CENTRE = np.array([0.016, 0.01225, 0.012])  # [m]
CAMERA_DISTANCE = 0.5  # [m] above the RFA top
HOUSING_SURFACE = CASE_DIR / "data" / "housing_surface.vtp"  # run make_housing_scene.py first

config_optical = {
    "camera": {
        "type": "pinhole",  # no calibration yet; use "opencv" or "vectors" when available
        "position": list(RFA_TOP_CENTRE + [0.0, 0.0, CAMERA_DISTANCE]),  # [m]
        "look_at": list(RFA_TOP_CENTRE),  # [m], centre of the image
        "up": [0.0, 1.0, 0.0],  # +y (long side of the RFA) points up in the image
        # Small field of view: 2 deg covers ~17.5 mm at the RFA top, 0.17 mm per pixel
        "fov": 2.0,  # [deg]
        "pixels": [100, 100],  # (nx, ny)
    },
    # Target (RFA) material: SiO2-like grey body as in the thermal model. Replace by the real
    # RFA material when known. The apparent temperature assumes the target material's
    # emissivity (0.8 here).
    "materials": {"SiO2": {"type": "grey", "emissivity": 0.8, "reflection": "diffuse"}},
    "surface_material": "SiO2",
    # Housing around the RFA: outer surfaces of housing_1 and housing_2, built by
    # make_housing_scene.py. Tungsten (built-in material, reflects like a mirror) at a constant
    # 373 K, its fixed-temperature boundary and initial temperature in the thermal model.
    "scene": [
        {"mesh": str(HOUSING_SURFACE), "material": "tungsten", "temperature": 373.0},
    ],
    "max_reflections": 5,
    "wavelength_range": [3000.0, 5000.0],  # [nm], mid-wave IR band
    "pixel_samples": 10,
    "spectral_bins": 20,
}

optical_case = OpticalModel(
    name="cea_mousetrap_ir_camera", enabled=True, backend_code="raysect", config=config_optical
)

irflow_case = IRFlow(
    case_name="CEA_mousetrap_openfoam_ir_images",
    root_dir="/home/brankm/data/irflow_studies/",
    steps=[flt_case, tm_case, optical_case],
    overwrite=True,
)

irflow_case.run()
