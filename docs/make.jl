using Documenter, DocumenterMarkdown, Lux, Pkg

using PythonCall, CondaPkg, Pkg  # Load mkdocs dependencies

deployconfig = Documenter.auto_detect_deploy_system()
Documenter.post_status(deployconfig; type="pending", repo="github.com/LuxDL/Lux.jl.git")

makedocs(; sitename="Lux", authors="Avik Pal et al.", clean=true, doctest=true,
    modules=[Lux], checkdocs=:all, format=Markdown(), draft=false,
    strict=[:doctest, :linkcheck, :parse_error, :example_block, :missing_docs],
    build=joinpath(@__DIR__, "docs"))

Pkg.activate(@__DIR__)

CondaPkg.withenv() do
    current_dir = pwd()
    cd(@__DIR__)
    run(`mkdocs build`)
    cd(current_dir)
    return
end

deploydocs(; repo="github.com/LuxDL/Lux.jl.git", push_preview=true, target="site",
    devbranch="main")
