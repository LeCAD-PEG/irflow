# examples/minimal_dummy_flt.py
from pathlib import Path

from irflow.core.flow import IRFlow
from irflow.core.registry import BackendRegistry
from irflow.core.executor import LocalExecutor
from irflow.steps.thermal_model import ThermalModel
from irflow.steps.therm import FieldLineTracing
from irflow.models.tm_config import ThermalModelConfig
from irflow.backends.field_line_tracing.dummy import DummyFLTBackend


registry = BackendRegistry()
registry.register("flt", "Dummy", DummyFLTBackend)

flow = IRFlow(
    case_name="thermal_model_001",
    root_dir=Path("./runs"),
    executor=LocalExecutor(),
    registry=registry,
    metadata={"purpose": "dummy FLT test"},
)

tl = ThermalModel(
    config=ThermalModelConfig(
        backend="ElmerFEM",
        mesh_file="kvader_tungsten_steel",
        mesh_type="elmer_mesh",
        sif_file="/home/brankm/work/heatfem/kvader_tungsten_steel"
    )
)


flow.add_step(tl)
result = flow.run()

print("Case dir:", result.case_dir)
print("Exchange:", result.step_results[-1].exchange_h5_path)
