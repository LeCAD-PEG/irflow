# External codes

Source repositories of codes used by irflow, kept as git submodules:

| Directory | Code | Version | Used by |
|---|---|---|---|
| `io_utils` | io_utils (git@github.com:mbrank/io_utils.git) | `main` | no longer imported by irflow |
| `raysect` | [raysect](https://github.com/raysect/source) ray tracer | tag `v0.8.1` | optical model backend `raysect` |

Fetch them after cloning irflow (shallow, raysect's history is large):

```bash
git submodule update --init --depth 1
```

## raysect

irflow does not build or install anything from `external/`: it imports whatever `raysect` is
importable in your Python environment. Check which one that is with

```bash
python -c "import raysect; print(raysect.__version__, raysect.__file__)"
```

The source here is also the API reference: the `.pyx` / `.pxd` files show classes and methods
that the compiled package hides (e.g. `raysect/primitive/mesh/mesh.pxd` for `MeshIntersection`).

If raysect is not installed, build it from this checkout (needs a C compiler; the build
installs Cython and numpy itself):

```bash
cd external/raysect
pip install .
```

For development, `dev/build.sh` builds the extensions in place instead; then put
`external/raysect` on `PYTHONPATH`. Keep the installed version and the submodule tag the same.
