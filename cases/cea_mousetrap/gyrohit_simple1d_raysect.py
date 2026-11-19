"""CEA mousetrap: the whole chain, from heat flux to synthetic IR camera images.

GYROHIT heat flux -> simple1d thermal model -> raysect camera. Needs no external solver, so it
runs in about a minute. The result is one image per thermal time step in
``<case>/OpticalModel/images.pvd`` (open it in ParaView as a time series).
"""

from irflow import FieldLineTracing, IRFlow, OpticalModel, ThermalModel

DATA_DIR = "/home/brankm/work/CEA_mousetrap/"

config_flt = {"target_mesh_path": DATA_DIR + "target.vtk", "target_mesh_type": "vtk"}

flt_case = FieldLineTracing(
    name="gyrohit_cea_mousetrap", enabled=True, backend_code="gyrohit", config=config_flt
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
    # One camera image per thermal time step: 21 images from 0 to 0.2 s
    "time_stepping": {"type": "transient", "time_step": 0.01, "end_time": 0.2},
    "n_points": 51,
}

tm_case = ThermalModel(
    "cea_mousetrap_simple1d", enabled=True, backend_code="simple1d", config=config_tm
)

# The sample is the box 0.015 < x < 0.017, 0.00725 < y < 0.01725, 0 < z < 0.012 [m], heated on
# its x = 0.015 face. The camera looks at that face from 65 mm in front of it, slightly above.
# No calibration of the real camera is available yet, so it is an ideal pinhole camera.
# The SiO2 sample is a grey diffuse body; its apparent temperature assumes its emissivity 0.8.
config_optical = {
    "camera": {
        "type": "pinhole",
        "position": [-0.05, 0.0, 0.025],  # [m]
        "look_at": [0.015, 0.01225, 0.008],  # [m], centre of the image
        "up": [0.0, 0.0, 1.0],  # z points up in the image
        "fov": 16.0,  # [deg] across the longer image side
        "pixels": [320, 256],  # (nx, ny)
    },
    "materials": {"SiO2": {"type": "grey", "emissivity": 0.8, "reflection": "diffuse"}},
    "surface_material": "SiO2",
    "wavelength_range": [3000.0, 5000.0],  # [nm], mid-wave IR band
    "pixel_samples": 10,
    "spectral_bins": 20,
}

optical_case = OpticalModel(
    name="cea_mousetrap_ir_camera", enabled=True, backend_code="raysect", config=config_optical
)

irflow_case = IRFlow(
    case_name="CEA_mousetrap_ir_images",
    root_dir="/home/brankm/data/irflow_studies/",
    steps=[flt_case, tm_case, optical_case],
    overwrite=True,
)

irflow_case.run()
