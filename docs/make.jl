using Documenter, DocumenterVitepress, DocumenterCitations
using Lux, LuxCore, LuxLib, WeightInitializers, NNlib
using LuxTestUtils, MLDataDevices

using Optimisers # for some docstrings

deploy_config = Documenter.auto_detect_deploy_system()
deploy_decision = Documenter.deploy_folder(
    deploy_config;
    repo="github.com/LuxDL/Lux.jl",
    devbranch="main",
    devurl="dev",
    push_preview=true,
)

makedocs(;
    sitename="Lux.jl Docs",
    authors="Avik Pal et al.",
    clean=true,
    doctest=false,  # We test it in the CI, no need to run it here
    modules=[Lux, LuxCore, LuxLib, WeightInitializers, LuxTestUtils, MLDataDevices, NNlib],
    linkcheck=true,
    linkcheck_ignore=[
        "http://www.iro.umontreal.ca/~lisa/publications2/index.php/attachments/single/205"
    ],
    repo="https://github.com/LuxDL/Lux.jl/blob/{commit}{path}#{line}",
    format=DocumenterVitepress.MarkdownVitepress(;
        repo="github.com/LuxDL/Lux.jl",
        devbranch="main",
        devurl="dev",
        deploy_url="https://lux.csail.mit.edu",
        deploy_decision,
    ),
    plugins=[CitationBibliography(joinpath(@__DIR__, "references.bib"))],
    draft=false
)

deploydocs(;
    repo="github.com/LuxDL/Lux.jl.git", push_preview=true, target="build", devbranch="main"
)
