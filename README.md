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
from irflow import IRFlow, FieldLineTracing, OpticalModel, ThermalModel

flt = FieldLineTracing(backend_code="riptrace", config={...})
tm = ThermalModel(backend_code="openfoam", config={...})  # or "elmerfem", "simple1d"
camera = OpticalModel(backend_code="raysect", config={...})
IRFlow(case_name="my_case", root_dir="~/irflow_studies", steps=[flt, tm, camera]).run()
```

Each step runs in `<root_dir>/<case_name>/<StepName>/` and saves its result there
(`save_formats=["vtk"]` by default): `FieldLineTracing/heat_flux.vtp`,
`ThermalModel/temperature.pvd` and `OpticalModel/images.pvd` (one IR image per thermal time
step). The `.pvd` files open directly in ParaView as time series.

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

The optical step (raysect) takes a camera, materials and optionally further scene components:

```python
config_optical = {
    # "pinhole" (position, look_at, up, fov, pixels), "opencv" (camera_matrix, distortion,
    # r_vector, t_vector, pixels) or "vectors" (file with per-pixel lines of sight)
    "camera": {"type": "pinhole", "position": [0, 0, 0.5], "look_at": [0, 0, 0],
               "up": [0, 1, 0], "fov": 2.0, "pixels": [100, 100]},
    "surface_material": "tungsten",  # default; physical (Fresnel + Kirchhoff emission)
    "materials": {"SiO2": {"type": "grey", "emissivity": 0.8, "reflection": "diffuse"}},
    "scene": [{"mesh": "housing.vtp", "material": "tungsten", "temperature": 373.0}],
}
```

Each image holds the band radiance and the apparent temperature, which assumes the emissivity
of the target (surface) material unless `assumed_emissivity` is given.

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
- `external/` — external code repositories (git submodules: `git submodule update --init
  --depth 1`), see `external/README.md`
- `archive/` — old code kept for reference only
