"""irflow: synthetic IR camera diagnostic workflow for magnetically confined fusion devices.

Typical use::

    from irflow import IRFlow, FieldLineTracing, OpticalModel, ThermalModel

    flt = FieldLineTracing(backend_code="riptrace", config={...})
    tm = ThermalModel(backend_code="openfoam", config={...})
    optical = OpticalModel(backend_code="raysect", config={...})
    IRFlow(case_name="my_case", root_dir="~/irflow_studies", steps=[flt, tm, optical]).run()
"""

from irflow.flow import IRFlow
from irflow.steps.field_line_tracing import FieldLineTracing
from irflow.steps.optical_model import OpticalModel
from irflow.steps.thermal_model import ThermalModel

__all__ = ["IRFlow", "FieldLineTracing", "ThermalModel", "OpticalModel"]
