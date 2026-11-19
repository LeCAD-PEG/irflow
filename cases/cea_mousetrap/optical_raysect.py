"""CEA mousetrap: synthetic IR camera images from a saved thermal model result.

Runs only the optical model (raysect), on the surface temperature series written by
``gyrohit_openfoam.py`` (``ThermalModel/temperature.pvd``); run that case first. The result is
one image per thermal time step in ``<case>/OpticalModel/images.pvd``.

The case gets its own name: with ``overwrite=True``, reusing ``CEA_mousetrap_transient`` would
delete the thermal model result that is the input here.
"""

from irflow import IRFlow, OpticalModel

STUDIES_DIR = "/home/brankm/work/irflow_studies/"
TEMPERATURE_FILE = STUDIES_DIR + "CEA_mousetrap_transient/ThermalModel/temperature.pvd"

# The sample is the box 0.015 < x < 0.017, 0.00725 < y < 0.01725, 0 < z < 0.012 [m], heated on
# its x = 0.015 face. The camera looks at that face from 65 mm in front of it, slightly above.
config_optical = {
    "camera_position": [-0.05, 0.0, 0.025],  # [m]
    "look_at": [0.015, 0.01225, 0.008],  # [m], centre of the image
    "up": [0.0, 0.0, 1.0],  # z points up in the image
    "fov": 16.0,  # [deg] across the longer image side
    "pixels": [320, 256],  # (nx, ny)
    "emissivity": 0.8,  # grey, constant
    "wavelength_range": [3000.0, 5000.0],  # [nm], mid-wave IR band
    "pixel_samples": 10,
    "spectral_bins": 20,
}

optical_case = OpticalModel(
    name="cea_mousetrap_ir_camera",
    enabled=True,
    backend_code="raysect",
    config=config_optical,
    input_data=TEMPERATURE_FILE,
)

irflow_case = IRFlow(
    case_name="CEA_mousetrap_transient_optical",
    root_dir=STUDIES_DIR,
    steps=[optical_case],
    overwrite=True,
)

irflow_case.run()
