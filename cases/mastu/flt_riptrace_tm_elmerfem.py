"""MAST-U: Riptrace field line tracing followed by a steady ElmerFEM thermal model."""

from pathlib import Path

from irflow import FieldLineTracing, IRFlow, ThermalModel

CASE_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = "/home/brankm/work/l2g/benchmarks/Test_MNEW8_sha12_res12/"

config = {
    "target_mesh_path": BENCHMARK_DIR + "target_res12_m.dat",
    "target_mesh_type": "dat",
    "shadow_mesh_path": BENCHMARK_DIR + "sha12_m_360deg.vtk",
    "shadow_mesh_type": "vtk",
    "equilibrium_path": {
        "ov5": BENCHMARK_DIR + "testov5.txt",
        "fiesta": BENCHMARK_DIR + "MNEW8.equ",
    },
    "equilibrium_type": "smardda",
    "power_loss": 1e6,  # Power in the scrape-off layer [W]
    "lambda_main": 0.01,  # main decay length [m]
    "point_omp": [1.3391667, 0],  # Location of separatrix/LCFS at outer midplane R, Z [m]
    "hpc": {"backend": "mpi", "omp_threads": 8},  # OpenMP tracing; omit for serial
}

flt_case = FieldLineTracing(
    name="riptrace_test_mastu", enabled=True, backend_code="riptrace", config=config
)

config_tm = {
    "mesh_file": str(CASE_DIR / "data" / "Mesh_1_scaled_m.unv"),
    "mesh_type": "unv",
    "sif_file": str(CASE_DIR / "case.sif"),
    "plasma_bc_id": 1,
    "time_stepping": {"type": "steady"},
}

tm_case = ThermalModel("mastu_elmerfem", enabled=True, backend_code="elmerfem", config=config_tm)

irflow_case = IRFlow(
    case_name="test_workflow_flt_and_tm",
    root_dir="/home/brankm/data/irflow_studies/",
    steps=[flt_case, tm_case],
    overwrite=True,
)

irflow_case.run()
