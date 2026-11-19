# IR flow
Workflow of IR camera synthetic diagnostic for magnetically confined fusion devices. The workflow connects three simulation steps:

1. field line tracing
2. thermal modelling
3. IR camera simulation

## Install

```bash
pip install -e ".[test]"
```

Backend codes are installed separately (see `external/`). Their locations on your machine go
into a code-paths file: copy `templates/code_paths.toml` to `~/.config/irflow/code_paths.toml`
(or point `$IRFLOW_CODE_PATHS` to it) and edit the paths.

## Use

```python
from irflow import IRFlow, FieldLineTracing, ThermalModel

flt = FieldLineTracing(backend_code="riptrace", config={...})
tm = ThermalModel(backend_code="openfoam", config={...})
IRFlow(case_name="my_case", root_dir="~/irflow_studies", steps=[flt, tm]).run()
```

Each step runs in `<root_dir>/<case_name>/<StepName>/` and saves its result there
(`save_formats=["vtk"]` by default): `FieldLineTracing/heat_flux.vtp` and
`ThermalModel/temperature.pvd` (a time series that opens directly in ParaView).

A step can also run on its own with data you provide:

```python
from irflow.io import load_heat_flux

heat_flux = load_heat_flux("riptrace_target.vtk", array_name="Q_cells")  # or a heat_flux.vtp
tm = ThermalModel(backend_code="elmerfem", config={...}, input_data=heat_flux)
IRFlow(case_name="tm_only", root_dir="~/irflow_studies", steps=[tm]).run()
```

Before solving, the thermal model checks that its plasma surface lies on the heat flux
surface (catches m/mm mix-ups and shifted meshes); set `mapping_tolerance` [m] in its config
to change the allowed distance.

Configs are plain dicts. They are checked when a step is created, before anything runs:
missing keys, misspelled keys (with a "did you mean ..." hint) and bad `time_stepping`
values raise a `ValueError` listing all problems.

Complete examples are in `cases/`.

## Test

```bash
pytest
```

## Layout

- `src/irflow/` — the package: `flow.py` (IRFlow), `steps/`, `backends/`
- `cases/` — study drivers (large input data in `cases/*/data/` is not tracked by git)
- `templates/` — configuration templates
- `tests/` — pytest tests
- `external/` — external code repositories (git submodules)
- `archive/` — old code kept for reference only
