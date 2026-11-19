# irflow — guidance for Claude

## Standards

Prioritize readability and simplicity over cleverness and complexity.
**Ask clarifying questions before making architectural changes.**

- Type hints required on all functions
- pytest for testing (fixtures in `tests/conftest.py`)
- PEP 8 with 100 character lines (`ruff check` uses the config in `pyproject.toml`)
- No `pdb.set_trace()` / `breakpoint()` in committed code
- Never write outputs to the current working directory; always into the step directory
- Commit only when the user asks

Full physics/requirements description: `CONTEXT_FOR_CLAUDE.md`.

## Project in one paragraph

irflow chains three simulation steps into a synthetic IR-camera diagnostic for tokamaks
(WEST, ITER), to be compared against experimental images:

1. **Field line tracing (FLT)** → heat flux on a triangular target mesh (per triangle).
   Backends: Riptrace (primary, in-house, Python lib `riptrace_py`), L2G, GYROHIT.
2. **Thermal model (TM)** → temperature on the plasma-facing surface (per node).
   Backends: OpenFOAM (primary, `foamMultiRun` multi-region solid), ElmerFEM, HeatFEM.
3. **Optical simulation** → 2D IR camera image. Ray tracing first (raysect primary, optika);
   a radiosity algorithm may be added later with the same inputs/outputs.

`IRFlow` takes a list of step objects; each step picks a backend from its config, runs it in
`<root_dir>/<case_name>/<step.dir_name>/` and hands its result to the next step.

## Decisions (agreed with user)

- **Time series:** transient TM → compare a series of IR images, one per time step.
  Steady TM → one image. TM output therefore always carries `times` (Nt,) and
  `temperature` (Nt, N); steady has Nt = 1.
- **Saved data format:** VTK now. HDF5 / MED must be easy to add later; the user script
  chooses the format (e.g. `save_to_vtk` / `save_to_hdf5` / `save_to_med`, or writer classes).
  Do not hard-wire VTK into the step classes.
- **Machine-specific paths** (OpenFOAM bashrc, Elmer bin dir) live in a TOML code-paths file
  (`irflow.code_paths`), not in case configs. Lookup: explicit path → `$IRFLOW_CODE_PATHS` →
  `~/.config/irflow/code_paths.toml`. Template: `templates/code_paths.toml`.
- **Old AI scaffold** is in `archive/irflow_scaffold/` — inspiration only, never import it.
- Optical step class name proposal: `OpticalModel` (`steps/optical_model.py`) — not yet confirmed.
- **CEA mousetrap** is a melting test: very high surface temperatures (~9000 K) are expected.
  For now the goal is that cases run; the physical setup may be revisited later.

## Dev environment

- User's Python env: venv `/home/brankm/work/workpython` (Python 3.10; numpy, scipy,
  pyvista, riptrace_py, raysect, h5py, tomli). irflow is installed there in editable mode
  with the `test` extra (pytest).
- Run tests: `/home/brankm/work/workpython/bin/python -m pytest` from repo root.
- Lint: `ruff check src tests cases`; types: `mypy` (both in `~/.local/bin`).
- OpenFOAM-dev at `/home/brankm/work/OpenFOAM-dev` (bashrc in `~/.config/irflow/code_paths.toml`);
  ElmerGrid/ElmerSolver on PATH. Disk is nearly full — keep test runs short and clean them up.
- Work happens on branch `package-restructure` (created 2026-11-19), not on `main`.

## Code map

| Path | Role |
|---|---|
| `src/irflow/flow.py` | `IRFlow`: case dir (`overwrite=False` by default), runs steps, passes data |
| `src/irflow/code_paths.py` | machine config for external codes |
| `src/irflow/steps/field_line_tracing.py` | `FieldLineTracing`; backends imported lazily |
| `src/irflow/steps/thermal_model.py` | `ThermalModel`; backends imported lazily |
| `src/irflow/backends/flt/riptrace.py` | Riptrace in-process via `riptrace_py` |
| `src/irflow/backends/flt/gyrohit.py` | reads precomputed GYROHIT heat flux from VTK (pyvista) |
| `src/irflow/backends/tm/openfoam/` | `backend.py` (run + read time series), `case_builder.py` (dict writers), `templates.py` + `template_files/` (coded BC, shell header) |
| `src/irflow/backends/tm/elmerfem.py` | sif parser/writer, UNV→Elmer mesh, per-triangle BCs |
| `cases/` | study drivers; large data in `cases/*/data/` (git-ignored) |
| `tests/` | pytest suite (no external codes needed) |
| `external/io_utils` | git submodule (user's io_utils); irflow no longer imports it |

### Data passed between steps (Phase 1, still plain dicts)

- FLT → TM: `{"xyz": (M, 3) cell centres, "heat_flux": (M,) W/m²}`
- TM → optical: `{"nodes": (N, 3), "cells_2d": (M, 3), "times": (Nt,), "temperature": (Nt, N) K}`

## Known issues (open)

- ElmerFEM: one BC per plasma triangle (won't scale); output reader is unverified, reads
  ElmerPost-style text from `elmer_temperatures.vtk` with plasma group name `"htc1"` hardcoded
  and only one time; `read_sif` relies on block headers appearing before keys.
- OpenFOAM coded BC: brute-force IDW over all FLT points per face per timestep; computes `k(T)`
  itself and only supports constant `kappa` or `kappaCoeffs<8>` — `table` heat conductivity
  would make the solver stop with FatalError.
- `case_builder.write_physicalProperties`: without `material_properties` it silently uses
  SiO2-like defaults; `molWeight` hardcoded.
- `case_builder.py` and `elmerfem.py` still have >100-char lines and style issues; they are
  left for the templating rework (Phase 4).
- No check yet that FLT target and TM plasma surface overlap (unit/translation mistakes are
  silent).

## Plan and status

**Phase 0 — Hygiene** ✅ (2026-11-19)
`.gitignore`; build artifacts / caches / backups untracked; `pdb` calls removed; `io_utils` is a
proper submodule in `external/`; old scaffold moved to `archive/`.

**Phase 1 — Package + vertical slice fixes** ✅ (2026-11-19)
Root `pyproject.toml`, src-layout package, drivers in `cases/`, code-paths config, bug fixes
(Riptrace import, OpenFOAM `get_temperature` with full time series, TM pass data, outputs no
longer written to CWD, Elmer `bc_id` parsing for ids ≥ 10), 30 tests.

**Phase 2 — Contracts** (next)
1. `data.py`: `SurfaceMesh`, `HeatFluxField`, `TemperatureField` dataclasses replacing dicts.
2. Writers chosen in the user script (VTK now; HDF5/MED later) — each step saves its result.
3. Step input override (`input=` object or file) so steps can run alone.
4. `backends/base.py` ABC; adapt Riptrace, Gyrohit, OpenFOAM, Elmer.
5. `mapping.py` with overlap/distance check (fail above tolerance).
6. Config dicts → per-backend dataclasses with validation (users may still pass dicts).

**Phase 3 — End-to-end with primary backends:** Riptrace → OpenFOAM → raysect
1. `OpticalModel` step + raysect backend (camera geometry, emissivity, spectral band), one
   image per TM time.
2. One reference case producing IR images.

**Phase 4 — Breadth and robustness**
1. ElmerFEM: single plasma BC + interpolation function instead of per-triangle BCs.
2. OpenFOAM: Python-side interpolation + stock BC; dictionaries from template files.
3. L2G, HeatFEM, optika backends; radiosity option for the optical step.
4. Golden-file tests for writers; integration tests that skip when a solver is missing.

## Open questions (ask the user)

- Optical inputs beyond the context file: emissivity / reflectivity per material, camera
  spectral band, and whether the optical scene mesh (for reflections) differs from the TM
  surface.
- Where large meshes for `cases/` live long-term (outside repo, git-lfs, data server).
