"""
This script builds the Pollen.jl documentation so that it can be loaded
by the frontend. It accepts one argument: the path where the generated
files should be stored.

    > julia docs/make.jl DIR

Use `./serve.jl` for interactive development.
"""

# Create target folder
isempty(ARGS) && error("Please pass a file path to make.jl:\n\t> julia docs/make.jl DIR ")
DIR = abspath(mkpath(ARGS[1]))

# Create Project
project = include("project.jl")

@info "Rewriting documents..."
Pollen.rewritesources!(project)

@info "Writing to disk at \"$DIR\"..."
Pollen.build(
    FileBuilder(
        JSONFormat(),
        DIR,
    ),
    project,
)
