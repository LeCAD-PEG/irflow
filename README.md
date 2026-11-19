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
