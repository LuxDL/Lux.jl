using Documenter, DocumenterMarkdown, Lux, Pkg

import Flux  # Load weak dependencies

using PythonCall, CondaPkg, Pkg  # Load mkdocs dependencies

deployconfig = Documenter.auto_detect_deploy_system()
Documenter.post_status(deployconfig; type="pending", repo="github.com/LuxDL/Lux.jl.git")

makedocs(; sitename="Lux", authors="Avik Pal et al.", clean=true, doctest=true,
         modules=[Lux],
         strict=[:doctest, :linkcheck, :parse_error, :example_block, :missing_docs],
         checkdocs=:all, format=Markdown(), draft=false, build=joinpath(@__DIR__, "docs"))

Pkg.activate(@__DIR__)

CondaPkg.withenv() do
    current_dir = pwd()
    cd(@__DIR__)
    run(`mkdocs build`)
    cd(current_dir)
end

deploydocs(; repo="github.com/LuxDL/Lux.jl.git", push_preview=true, target="site",
           devbranch="main")
