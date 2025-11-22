using Documenter, DocumenterVitepress, DocumenterCitations
using Lux, LuxCore, LuxLib, WeightInitializers, NNlib
using LuxTestUtils, MLDataDevices

using Optimisers # for some docstrings

const DRAFT_MODE = parse(Bool, get(ENV, "LUX_DOCS_DRAFT_BUILD", "false"))
if DRAFT_MODE
    @info "Building docs in DRAFT mode. This is intended for testing only."
    # In draft mode. This is only done locally, so we trigger the tutorial as well.
    withenv("LUX_DOCUMENTATION_NTASKS" => 12) do
        include("tutorials.jl")
    end
end

warnonly = [:linkcheck]
if DRAFT_MODE
    push!(warnonly, :cross_references)
end

makedocs(;
    sitename="Lux.jl Docs",
    authors="Avik Pal et al.",
    clean=DRAFT_MODE,
    doctest=false,  # We test it in the CI, no need to run it here
    modules=[Lux, LuxCore, LuxLib, WeightInitializers, LuxTestUtils, MLDataDevices, NNlib],
    warnonly,
    linkcheck_ignore=[],
    repo="https://github.com/LuxDL/Lux.jl/blob/{commit}{path}#{line}",
    format=DocumenterVitepress.MarkdownVitepress(;
        repo="github.com/LuxDL/Lux.jl",
        devbranch="main",
        devurl="dev",
        deploy_url="https://lux.csail.mit.edu",
    ),
    plugins=[CitationBibliography(joinpath(@__DIR__, "references.bib"))],
    draft=DRAFT_MODE,
)

DocumenterVitepress.deploydocs(;
    repo="github.com/LuxDL/Lux.jl.git", push_preview=true, target="build", devbranch="main"
)
