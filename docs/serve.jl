"""
This script serves the Pollen.jl documentation on a local file server
so that it can be loaded by the frontend in development mode.
files should be stored.

    > julia docs/serve.jl

Use `./make.jl` to export the generated documents to disk.

There are two modes for interactive development: Lazy and Regular.
In lazy mode, each document will be built only if it is requested in
the frontend, while for Regular mode, each document will be built
once before serving.
"""

using Pollen

project = include("project.jl")


Pollen.serve(
    project;
    lazy = get(ENV, "POLLEN_LAZY", "false") == "true",
    port = Base.parse(Int, get(ENV, "POLLEN_PORT", "8000")),
    format = JSONFormat()
)
