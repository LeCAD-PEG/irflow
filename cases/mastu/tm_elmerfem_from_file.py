"""MAST-U: steady ElmerFEM thermal model on its own, with heat flux from an earlier Riptrace run.

Shows how to run one step with your own input data instead of the previous step's result.
"""

from pathlib import Path

from irflow import IRFlow, ThermalModel
from irflow.io import load_heat_flux

CASE_DIR = Path(__file__).resolve().parent
RIPTRACE_TARGET = (
    "/home/brankm/work/irflow_studies/test_workflow_flt_only/"
    "FieldLineTracing/riptrace_case/smardda_caseirflow_target.vtk"
)

# Riptrace stores the heat flux as cell array "Q_cells"; a heat_flux.vtp saved by irflow
# could be passed directly as input_data="path/to/heat_flux.vtp".
heat_flux = load_heat_flux(RIPTRACE_TARGET, array_name="Q_cells")

config_tm = {
    "mesh_file": str(CASE_DIR / "data" / "Mesh_1_scaled_m.unv"),
    "mesh_type": "unv",
    "sif_file": str(CASE_DIR / "case.sif"),
    "plasma_bc_id": 1,
    "time_stepping": {"type": "steady"},
}

tm_case = ThermalModel(
    "mastu_elmerfem", backend_code="elmerfem", config=config_tm, input_data=heat_flux
)

irflow_case = IRFlow(
    case_name="test_workflow_tm_only",
    root_dir="/home/brankm/work/irflow_studies/",
    steps=[tm_case],
    overwrite=True,
)

irflow_case.run()
