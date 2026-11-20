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
   Backends: OpenFOAM (primary, `foamMultiRun` multi-region solid), ElmerFEM, HeatFEM, and
   `simple1d` (fast estimate: independent 1D slab under every heat flux triangle).
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
- Optical step is `OpticalModel` (`steps/optical_model.py`); name proposed by Claude, not
  objected to by the user.
- **simple1d:** first grid cell chosen automatically from the time step
  (`AUTO_FIRST_CELL_FACTOR` × sqrt(α dt), `first_cell_size` overrides); backward Euler accuracy
  accepted; nodes used by no triangle get a constant `unused_node_temperature` (default 110 K —
  user said "around 110", unit assumed K).
- **Optical:** calibrated camera model; output apparent temperature (besides radiance); flat
  spectral band kept; emissivity per material with tungsten as default; reflections required.
- **CEA mousetrap** is a melting test: very high surface temperatures (~9000 K) are expected.
  For now the goal is that cases run; the physical setup may be revisited later.

## Dev environment

- User's Python env: venv `/home/brankm/work/workpython` (Python 3.10; numpy, scipy,
  pyvista, riptrace_py, raysect, h5py, tomli). irflow is installed there in editable mode
  with the `test` extra (pytest).
- Run tests: `/home/brankm/work/workpython/bin/python -m pytest` from repo root.
- Lint: `ruff check src tests cases`; types: `mypy` (both in `~/.local/bin`).
- OpenFOAM-dev at `/home/brankm/work/OpenFOAM-dev` (bashrc in `~/.config/irflow/code_paths.toml`);
  ElmerGrid/ElmerSolver on PATH.
- **Disk:** the root disk is nearly full. Study outputs live on the data disk:
  `/home/brankm/data/irflow_studies` (cases' `root_dir`; `/home/brankm/work/irflow_studies` is a
  symlink to it). Claude's verification runs go to `/home/brankm/data/irflow_verification`,
  never into `irflow_studies` (cases use `overwrite=True`).
- Work happens on branch `package-restructure` (created 2026-11-19), not on `main`.

## Code map

| Path | Role |
|---|---|
| `src/irflow/flow.py` | `IRFlow`: case dir (`overwrite=False`), runs steps, saves results (`save_formats`), passes data |
| `src/irflow/data.py` | `SurfaceMesh`, `HeatFluxField`, `TemperatureField` (validated, SI units) |
| `src/irflow/io/` | `save()` / `load_heat_flux()` / `load_temperature()`; `vtk.py` (`.vtp`, `.pvd` series) |
| `src/irflow/mapping.py` | `check_overlap`: TM plasma surface must lie on the FLT surface |
| `src/irflow/code_paths.py` | machine config for external codes |
| `src/irflow/config.py` | `check_config` (missing/unknown keys, typo hints), `check_time_stepping` |
| `src/irflow/steps/field_line_tracing.py` | `FieldLineTracing`; backends imported lazily |
| `src/irflow/steps/thermal_model.py` | `ThermalModel`; `input_data=` to run alone; `mapping_tolerance` config |
| `src/irflow/backends/base.py` | ABCs `FieldLineTracingBackend`, `ThermalModelBackend` (`run`, `get_result`) |
| `src/irflow/backends/flt/riptrace.py` | Riptrace in-process via `riptrace_py` (`riptrace_case`, `riptrace_mesh`, `io_reader`) |
| `src/irflow/backends/flt/gyrohit.py` | reads precomputed GYROHIT heat flux from VTK |
| `src/irflow/backends/tm/openfoam/` | `backend.py` (run, pre-solve overlap check, read time series), `case_builder.py` (fills dictionary templates), `templates.py` + `template_files/` (`@NAME@` placeholders) |
| `src/irflow/backends/tm/openfoam/polymesh.py` | read outer patches of an ASCII `polyMesh` as `SurfaceMesh` (scene components) |
| `src/irflow/backends/tm/simple1d.py` | 1D slab per FLT triangle; steady analytic, transient backward Euler (one LU for all slabs), optional graded grid `first_cell_size` |
| `src/irflow/steps/optical_model.py` | `OpticalModel`; `input_data=` (TemperatureField or `temperature.pvd`) |
| `src/irflow/backends/optical/raysect.py` | backend: scene (TM surface + `scene` components), render one image per time, apparent temperature |
| `src/irflow/backends/optical/raysect_cameras.py` | cameras `pinhole` (`RadianceCamera`), `opencv` and `vectors` (`LineOfSightCamera`, rays fill each pixel exactly, weight 1) |
| `src/irflow/backends/optical/raysect_materials.py` | material library (`tungsten`: Fresnel from raysect n,k + Kirchhoff emission; `grey`: diffuse/specular/none), node/constant temperature sources |
| `src/irflow/backends/optical/planck.py` | Planck radiance, band integral (raysect's midpoint bins), apparent-temperature inversion |
| `src/irflow/backends/tm/elmerfem.py` | sif parser/writer, UNV→Elmer mesh, per-triangle BCs, pyvista output reader |
| `cases/` | study drivers; large data in `cases/*/data/` (git-ignored) |
| `tests/` | pytest suite (no external codes needed) |
| `external/io_utils` | git submodule (user's io_utils); irflow no longer imports it |
| `external/raysect` | git submodule, raysect source at tag v0.8.1 (reference; irflow imports the installed raysect) |

### Data passed between steps (`irflow.data`)

- FLT → TM: `HeatFluxField(mesh, heat_flux (M,) W/m²)` — saved as `FieldLineTracing/heat_flux.vtp`
- TM → optical: `TemperatureField(mesh, times (Nt,), temperature (Nt, N) K)` — saved as
  `ThermalModel/temperature.pvd` (+ `temperature/temperature_NNNN.vtp`)
- Optical output: `CameraImages(times, radiance (Nt, ny, nx) W/(m² sr), wavelength_range nm,
  apparent_temperature (Nt, ny, nx) K, assumed_emissivity)`, row 0 = image top, OpenCV
  (u, v) = (column, row) — saved as `OpticalModel/images.pvd` (+ `images/image_NNNN.vti`, rows
  flipped so ParaView shows them upright). Apparent T is NaN below 50 K (e.g. pixels that see
  nothing).
- A step with user `input_data` has `has_own_input = True`; IRFlow then does not overwrite it.
- Overlap check default tolerance: 0.5 × longest edge of either mesh; distance is measured from
  TM face centres to the FLT *surface* (resolution independent).

## Known issues (open)

- ElmerFEM: one BC per plasma triangle (won't scale; MAST-U case has 6180 BCs);
  `read_sif` relies on block headers appearing before keys; mesh_type `"mesh"` path does not
  copy `mesh.header`/`mesh.names` into the step dir (untested). Elmer needs short paths —
  `Mesh DB` is written relative to the step dir for that reason.
- Riptrace: current `riptrace_py` has no `Shadow.load_dat`, so only VTK shadow meshes work.
- OpenFOAM coded BC: brute-force IDW over all FLT points per face per timestep; computes `k(T)`
  itself and only supports constant `kappa` or `kappaCoeffs<8>` — `table` heat conductivity
  would make the solver stop with FatalError.
- `case_builder.write_physicalProperties`: without `material_properties` it silently uses
  SiO2-like defaults; `molWeight` hardcoded. The OpenFOAM model names written for table /
  polynomial properties (`DENSITY_MODELS` etc. in `case_builder.py`) look inconsistent
  (e.g. `hPolynomial` as solid transport, `eIcoTabulated` as equation of state) and were never
  run; only constant properties are verified. Kept byte-identical on purpose (golden tests).
- `elmerfem.py` still has >100-char lines, style issues and mypy errors (rework in Phase 4).
- VTK temperature series writes the full mesh for every time: mousetrap simple1d, 201 times on
  132k triangles = 241 MB and ~70% of the run time (user: fine, outputs are on the data disk).
- raysect backend: materials are Python (~60 µs per reflection per core); tungsten uses
  smooth room-temperature n,k (Rakić 1998) → ε ≈ 0.023–0.034 in 3–5 µm, far below measured
  hot/rough PFC tungsten (~0.1–0.3); scene components have constant temperature and must not
  overlap the TM surface; only the 5-coefficient OpenCV distortion model; no background
  radiation (escaping rays give 0); diffuse reflections of small hot spots are noisy
  (1/sqrt(pixel_samples)); no detector response, filter or transmission.
- `ruff check` without `--select` skips line length (E501); use
  `ruff check --select E,F,W,I src tests cases`.

## Plan and status

**Phase 0 — Hygiene** ✅ (2026-11-19)
`.gitignore`; build artifacts / caches / backups untracked; `pdb` calls removed; `io_utils` is a
proper submodule in `external/`; old scaffold moved to `archive/`.

**Phase 1 — Package + vertical slice fixes** ✅ (2026-11-19)
Root `pyproject.toml`, src-layout package, drivers in `cases/`, code-paths config, bug fixes
(Riptrace import, OpenFOAM `get_temperature` with full time series, TM pass data, outputs no
longer written to CWD, Elmer `bc_id` parsing for ids ≥ 10), 30 tests.

**Phase 2 — Contracts** ✅ (2026-11-19)
1. ✅ `data.py`: `SurfaceMesh`, `HeatFluxField`, `TemperatureField` dataclasses replacing dicts.
2. ✅ `IRFlow(save_formats=["vtk"])` + `irflow.io` (option (a) chosen by user).
3. ✅ `ThermalModel(input_data=...)` so the thermal model can run alone.
4. ✅ `backends/base.py` ABCs; Riptrace, Gyrohit, OpenFOAM, Elmer adapted.
5. ✅ `mapping.py` overlap check (OpenFOAM: exports plasma patch at t=0 before solving).
6. ✅ Config validation with plain dicts (`irflow.config`, user chose this over dataclasses):
   missing/unknown keys with typo hints, time stepping, OpenFOAM BC types — at step creation.

Verified by real runs (scratch dir): Gyrohit→OpenFOAM mousetrap (10 steps, identical T to
Phase 1), ElmerFEM alone on MAST-U with old Riptrace `Q_cells` (T identical to the March run),
Riptrace MAST-U benchmark with current `riptrace_py` (heat flux matches the earlier run; serial
tracing ~10 min, OpenMP `"hpc": {"backend": "mpi", "omp_threads": 8}` 32 s, identical result).

**Phase 3 — End-to-end with primary backends:** Riptrace → OpenFOAM → raysect
1. ✅ `OpticalModel` step + raysect backend, one image per TM time (subagent, merged
   2026-11-19). Checked: uniform plane within 0.2 % of ε × band Planck integral; linear T
   field reproduced inside triangles (test fails if barycentric weights are permuted).
2. ✅ `simple1d` thermal backend (subagent, merged 2026-11-19).
3. Reference cases producing IR images: `cases/cea_mousetrap/gyrohit_simple1d_raysect.py`
   (full chain, no external solver) and item 5; Riptrace → OpenFOAM → raysect (WEST) still to
   run — see "Instructions for an AI agent on an HPC machine" below.
4. ✅ Optical upgrade (subagent, merged 2026-11-19): calibrated cameras (OpenCV parameters,
   per-pixel vectors), apparent temperature (default assumed emissivity = target material's),
   material library with physical tungsten, grey materials, reflections, `scene` components.
   Checked: isothermal enclosure gives black body within 0.03 % for every material; mirror
   images and distorted cameras land where `cv2.projectPoints` puts them.
5. Case `cases/cea_mousetrap/gyrohit_openfoam_raysect.py` (user request): GYROHIT → OpenFOAM →
   raysect, own pinhole camera 0.5 m above the RFA top looking straight down, 100×100 px, 2°;
   `scene` = `cases/cea_mousetrap/data/housing_surface.vtp` (housing_1 + housing_2 outer
   surfaces, built by `make_housing_scene.py`), tungsten, 373 K; RFA grey SiO2-like ε 0.8.
   ✅ Ran end to end 2026-11-19 (16.5 min on the workstation: OpenFOAM ~15 min, 201 images
   ~1.5 min). Apparent peak 7597 K vs true edge peak 9010 K at 200 ms (0.175 mm pixels
   average the thin hot edge band); tungsten housing appears at 267 K instead of 373 K
   because the image is converted with the RFA's ε 0.8 (tungsten ε ≈ 0.026).

**Phase 4 — Breadth and robustness**
1. ElmerFEM: single plasma BC + interpolation function instead of per-triangle BCs.
2. OpenFOAM: Python-side interpolation + stock BC (physics change — ask first);
   ✅ dictionaries from template files, golden-file tests (`tests/test_openfoam_golden.py`,
   regenerate with `IRFLOW_UPDATE_GOLDEN=1` only for intended output changes).
3. L2G, HeatFEM, optika backends; radiosity option for the optical step.
4. Golden-file tests for writers; integration tests that skip when a solver is missing.

## Work log

**2026-11-19/20, workstation session (Claude + subagents in git worktrees):**

- Commits on `package-restructure` (oldest first): `edcc2ce` cleanup, `60164ee` src/irflow
  package + bug fixes, `e72d875` typed step data / saving / standalone steps / overlap check,
  `019d65e` config validation + OpenFOAM templates + Riptrace OpenMP, `89b0423` simple1d,
  `17d8faa` optical step (raysect), `789c7b9` simple1d automatic first cell, `c9f1f31` optical
  cameras / apparent temperature / materials / reflections, `cd6040e` GYROHIT → OpenFOAM →
  raysect RFA case with housing scene. 175 tests pass. `main` is still at `ff384ce`.
- Pushing: the default GitHub SSH key (`mbrank`, forced for github.com in `~/.ssh/config`)
  cannot write to `LeCAD-PEG/irflow`; the user said to use the `fsbrank` key. Working command:
  `GIT_SSH_COMMAND="ssh -F /dev/null -i $HOME/.ssh/fsbrankgithub -o IdentitiesOnly=yes
  -o AddressFamily=inet" git push`. `package-restructure` is on origin since 2026-11-20.
- Verified with real runs (outputs in `/home/brankm/data/irflow_verification`):
  Gyrohit → OpenFOAM mousetrap, full 0.2 s, 200 steps in 15 min; raysect images of it (oblique
  320×256: 201 images in 5.3 min; top view 100×100: 44 s); Riptrace MAST-U serial vs OpenMP
  (identical heat flux, 10 min vs 32 s); ElmerFEM MAST-U alone (T identical to the March run);
  Gyrohit → simple1d → raysect (21 images in 57 s).
- Study outputs moved from `/home/brankm/work/irflow_studies` to the data disk (symlink left).
- Fixed on the way: the mousetrap OpenFOAM case diverged ("negative initial temperature")
  with `Gauss linear corrected`; `Gauss linear limited corrected 0.33` (as in the user's
  last working manual run) is now written. Elmer's VTK writer crashed on long absolute
  `Mesh DB` paths (now relative). Riptrace modules were renamed in `riptrace_py`
  (`mag_riptrace_case` → `riptrace_case`, etc.); backend follows.
- Not done yet: commit/merge to `main`, Riptrace → OpenFOAM → raysect on WEST, realistic
  tungsten emissivity, deleting the old debug outputs in `src/` (`elmer_mesh/`, `*.vtk` —
  the user should delete them; the auto-mode check blocked Claude from doing it).

## Instructions for an AI agent on an HPC machine: large WEST case

Goal: run Riptrace → OpenFOAM (or simple1d first) → raysect on WEST divertor geometry, much
larger than the CEA mousetrap (one WEST lower-divertor target quarter alone has ~348k
triangles). Work in small, verified steps and report to the user after each one. Everything
in "Standards" above applies: ask before architectural or physics changes, commit only when
asked, never write into the repository or the home directory's small disk — use the cluster's
project/scratch filesystem for all `root_dir`s.

### 1. Environment

1. `git clone` the repo, `git checkout package-restructure`,
   `git submodule update --init --depth 1 external/raysect` (`external/io_utils` needs access
   to `mbrank/io_utils` and is not imported by irflow — skip it if it fails).
2. Python ≥ 3.10 venv (or the cluster's Python module), then `pip install -e ".[test]"`.
   raysect: `pip install raysect==0.8.1` or build `external/raysect` (`pip install .`);
   keep version 0.8.1 (the barycentric `MeshIntersection` API is used). `opencv-python` is
   needed only for `opencv` cameras.
3. `riptrace_py` is not on PyPI: clone the riptrace repo and build it following its README
   (`./compile_cpu.sh` for CPU/OpenMP, `./compile_alpaka.sh` with `CUDA_ARCH=...` for GPU),
   then make `import riptrace_py` work in the same venv. Check with
   `python -c "from riptrace_py.riptrace_case import RipTraceCase"`.
4. OpenFOAM: irflow needs `foamMultiRun` (OpenFOAM-dev / OpenFOAM 11+, openfoam.org) plus
   `ideasUnvToFoam`, `splitMeshRegions`, `foamToVTK`. Put its `etc/bashrc` into
   `~/.config/irflow/code_paths.toml` (template: `templates/code_paths.toml`) or set
   `$IRFLOW_CODE_PATHS`. If OpenFOAM comes as a module, a small wrapper bashrc that runs
   `module load ...` works.
5. `python -m pytest` must pass (175 tests, no external codes needed) before anything else.
6. Run the small chain `cases/cea_mousetrap/gyrohit_simple1d_raysect.py` with `root_dir` set to
   scratch, if its input `target.vtk` is copied over — it checks the whole install in ~1 min.

### 2. WEST input data (ask the user where it is on the cluster)

On the workstation it is in `/home/brankm/work/west` (git-LFS repo):

- `pfcflux_benchmark_2D/` and `benchmark/`: PFCFLUX benchmark. Targets
  `lo_div_graphite_q1a.dat`, `lo_div_graphite_q1b.dat` (Nastran, ~348k CTRIA3 each);
  shadowing meshes `DVT_Baffle_targets_*.dat`, `DVT_low_sect_front_*.dat`,
  `DVT_up_sect_front_*.dat`; equilibrium `benchmark/mag data/WEST_SN53585_t8.0.txt3d`
  (3D field + psi from TSbase, as used by PFCFLUX) and `pfcflux_benchmark_2D/WEST_500kA_Far.*`;
  PFCFLUX reference results in `benchmark/results`; the PFCFLUX setup script
  `west_collab_slovenie.py` shows how the benchmark was configured.
- `/home/brankm/work/WEST_thermal_modelling/openfoam_case`: an existing hand-made OpenFOAM
  case (not a UNV mesh) — ask the user which thermal mesh to use.

**Known traps with this data (check each, report, ask before changing code):**

- The `.dat` files are in **millimetres** (R ≈ 2064) and use **decimal commas**
  (`2064,053540070`) in long-format `GRID*` cards. irflow's Riptrace backend calls
  `Target.load_dat(path)` with riptrace's default `units="m"` — a `units` option has to be
  added to the backend/config (small change, but tell the user), and check whether riptrace's
  dat reader accepts decimal commas.
- Shadow meshes: irflow takes **one** shadow mesh path and only `"vtk"` (current riptrace has
  no `Shadow.load_dat`). Convert and merge the DVT shadowing `.dat` files into one VTK in metres
  first (a small script in `cases/west/`, like `make_housing_scene.py`).
- Equilibrium: use `"equilibrium_type": "pfcflux"` with `"equilibrium_path": {"txt": ...}`.
  riptrace's `read_pfcflux_data` has sign/scale options (`flip_b_phi`, `scale_*`) that irflow
  does not pass yet; compare the Riptrace heat flux against the PFCFLUX results before trusting
  it.
- `power_loss`, `lambda_main`, `point_omp` (R, Z of the separatrix at the outer midplane) are
  not documented anywhere in irflow for WEST: take them from the PFCFLUX script/results or
  ask the user. Never guess physics parameters silently.

### 3. Running at scale

- Run irflow inside a batch job on one node (`sbatch` with `--cpus-per-task` = cores),
  e.g. `srun python cases/west/<case>.py`. Log to a file on scratch.
- **Riptrace:** `"hpc": {"backend": "mpi", "omp_threads": <cores>}` (OpenMP in-process;
  "mpi" is riptrace's name for its threaded CPU backend). Single-GPU in-process backends
  (`"alpaka"`, `"alpaka3"`) may work if riptrace was built with them — untested through irflow.
  riptrace's multi-GPU / SLURM submission mode (`mode="slurm"`, `alpaka3_mgpu`; see the
  riptrace repo's `slurm_scale_west_*gpu.slm`) is **not wired into irflow** (the backend forces
  `mode="inprocess"`); ask the user before adding it. Start with a cut-down target (a few
  thousand triangles) to check the setup, then the full target.
- **Thermal model, recommended order:** `simple1d` first (vectorised, fine for 1e6 triangles,
  needs only the FLT mesh), then OpenFOAM. OpenFOAM limits to know before running it big:
  irflow runs `foamMultiRun` **serially** (no `decomposePar`/`mpirun`), and the coded plasma
  BC interpolates by brute force over **all** FLT points for **every** face at **every** time
  step — O(N_faces × N_FLT) per step, prohibitive for WEST-size meshes. Fixing this
  (Python-side interpolation + stock BC, parallel runs) is Phase 4 item 2 and changes the
  physics setup: propose it to the user, do not just do it. ElmerFEM (one BC per triangle)
  does not scale at all.
- The thermal mesh's plasma patch must lie on the FLT target in the same frame and in metres;
  `check_overlap` stops the run otherwise. Do not raise `mapping_tolerance` to silence it —
  find the unit/frame mismatch.
- **Output size:** every saved time rewrites the whole mesh (VTK). N_times × mesh size can
  reach many GB for WEST; use a coarser `time_step` or fewer steps, and write to scratch.
- **raysect:** set `"processes"` to the job's core count (the default uses all CPUs of the
  node, which oversubscribes shared nodes). Materials are Python (~60 µs per reflection per
  core): start with a small image (e.g. 64×64, `pixel_samples` 4, `max_reflections` 1),
  measure, then scale up. Tungsten from raysect's data has ε ≈ 0.026 (smooth, room
  temperature) — still an open question with the user; report it with any WEST image.
- No camera calibration exists for WEST yet: use a `pinhole` camera and say so; switch to
  `opencv` / `vectors` cameras when the user provides a calibration.

### 4. Reporting

After each step, report to the user: what ran, wall time, peak memory, the key numbers
(heat flux max/integral vs PFCFLUX, temperature range, image range), any warnings, and what
you changed in the code (with tests). Add a dated entry to "Work log" above.

## Open questions (ask the user)

- OpenFOAM model names for table/polynomial material properties (see known issues).
- Optical: calibration format when real data arrives (Calcam pixel-centre convention vs ours);
  tungsten emissivity far too low (rough surface via raysect `RoughConductor`, T-dependent n,k,
  or a tunable value?); `max_reflections` default 5 (closed tungsten enclosures need ~100);
  importance sampling of hot surfaces for diffuse reflections; per-region materials on the TM
  surface itself (needs region tags); scene components with time-dependent temperature.
- Where large meshes for `cases/` live long-term (outside repo, git-lfs, data server).
