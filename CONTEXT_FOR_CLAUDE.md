# Standards

When working with this codebase, prioritize readability and simplicity
over cleverness and complexity. Ask clarifying questions before making
architectural changes. Also:

- Type hints required on all functions
- pytest for testing (fixtures in `tests/conftest.py`)
- PEP 8 with 100 character lines
  
# Goal of project and current status

The goal of the irflow project is to prepare the workflow that
connects three simulation steps to arrive at synthetic diagnostics IR
image, which is compared to an experiment (in WEST and ITER
tokamaks). The simulation steps are:

- field line tracing
- thermal modelling
- optical simulation

## Simulation step 1: Field line tracing

Calculates power deposition to target (target is plasma facing
component). The codes we plan to use here are:

- Riptrace (primary): in-house code
- L2G (secondary): joint in-house and ITER code 
- GYROHIT (tertiary): in-house code, includes gyro radius simulation
  as well, currently struggles with larger meshes

Input to field line tracing step is:

- target mesh: triangular mesh in 3d space
- shadow mesh: triangular mesh in 3d space
- SOL power: single scalar value, representing power in the scrape-off layer
- Decay length of particles: single value
- OMP position: 2 single values, R and Z coordinate of separatrix/LCFS
  at outer midplane
- equilibrium and magnetic field data: for riptrace specifical input
  is:
  - one 3D segment of magnetic field from coils. This is structured
  grid with R,Phi,Z steps and vector components Br, Bphi, Bz. It is
  repeated N times to match 360 degress, i.e. 20 deg times 18 to reach
  360 deg.
  - 2D structured grid in R, Z plane of magnetic flux from plasma. In
    riptrace, it is converted to magnetic field and added to B from coils
  
Output of field line tracing step is:

- target mesh with heat fluxes triangular mesh in 3d space. Heat
  fluxes are defined on each triangle
  
## Simulation step 2: Thermal model

Calculates temperature from heat fluxes, calculated in step 1. The
codes we plan to use here are:

- OpenFOAM (primary): well known FVM solver
- ElmerFEM (secondary): well known FEM solver
- HeatFEM (tertiary): in-house code

Input to thermal model is:

- target mesh with heat fluxes from step 1.
- 3D model, where plasma facing surface overlaps with target mesh from
  step 1. Heat fluxes are mapped on the surface of 3D component as
  boundary condition
- other boundary conditions and material properties can also be
  defined as input, corresponding properly to 3D mesh volume and
  boundary groups

Output of thermal model is:

- Plasma facing surface with temperature at nodes (the surface is
  extracted from final result)

## Simulation step 3: Optical simulation

Calculates camera response from temperature map of plasma facing
surface in tokamak. This is primarily supposed to be a ray trasing
simulation. Later we might add radiosity algorithm, that should work
even better. But the input and output stays the same anyways. But keep
it in mind. The IR camera is looking into a reactor. The codes we plan
to use are:

- raysect (primary): well known fusion based ray-tracing tool, for
  scientists working in fusion
- optika (secondary): in-house code, it is being actively developed
  with specific purpose for synthetic IR camera images production
  
Input to optical simulation:

- IR camera geometry: Field of view, location and dimensions of sensor
  and aperture
- temperature map from simulation step 2

Output of optical simulation:

- 2D IR camera image, to be compared to experimental image


## Current status and plan

In folder `external` we should host repositories of the codes from the
simulation steps. However there should be a configuration file that
user can set up, in case codes are compiled in other folders (so that
they are not installed again in the folder)

In folder `src` is the source code. An aglorithmic design currently is:

One overall main file/class for the whole workflow. Currently in `flow.py`. Maybe
we should rename it to `irflow.py`. This is the main class. To this
class the user inputs the all the configuration files for the case.

Each simulation step has a general class:

- `field_line_tracing.py` has `FieldLineTracing`
- `thermal_model.py` has `ThermalModel`
- `optical_simulation.py` does not exist yet. I am not certain on the
  name yet. Maybe `optical_model.py`.
  
These classes should read the python configuration files for the case
and then use the proper backend (Riptrace, ElmerFEM, OpenFOAM,
Raysect, ...). The backend: 

- builds the case on command from general class (user's command in python should be)
- runs the case
- if next simulation step is waiting, it passes the data back through
  general class to the next simulation step


The user can also run just one simulation step, if he wants. For
second and third simulation step, there should be a configuration
option by default, that data is passed from previous step. But user
can also pass its own data in the apropriate format that corresponds
to current simulation step and code

Besides outputs/inputs to simulation steps, the user in the user's
python script (where he calls the main class) the user should also be
able to determine type of saved data. For now, let's stick to VTK, but
the option for hdf5/med shoudl easilly be added at some point. Maybe
savetovtk and savetomed and savetohdf5. Or respectable classes.

I propose following layout:

- `external`: external repos of codes, etc.
- `src`: source file
- `tests`: testing utilites and examples
- `examples`: basic examples, part of documentation
- `cases`: large scale cases, part of project. Don't yet know how to
  handle this. So let's keep it here for now.
- `templates`: this folder should contain templates for cases in
  advance, such as template configurations in python, maybe meshes?
  Some of them are already currently in the `src` folder and should be
  moved accordingly
  
In `src/irflow` is a complex example of the proposed workflow,
generated by old versions of AI. Can be examined for inspiration, but
will not be used in release version.


  
