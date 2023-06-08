using Documenter, DocumenterMarkdown, WeightInitializers

deployconfig = Documenter.auto_detect_deploy_system()
Documenter.post_status(deployconfig;
    type="pending",
    repo="github.com/LuxDL/WeightInitializers.jl.git")

makedocs(;
    sitename="WeightInitializers",
    authors="LuxDL contributors",
    clean=true,
    doctest=true,
    modules=[WeightInitializers],
    strict=[:doctest, :linkcheck, :parse_error, :example_block, :missing_docs],
    checkdocs=:all,
    format=Markdown(),
    draft=false,
    build=joinpath(@__DIR__, "docs"))

deploydocs(;
    repo="github.com/LuxDL/WeightInitializers.jl.git",
    push_preview=true,
    deps=Deps.pip("mkdocs",
        "pygments",
        "python-markdown-math",
        "mkdocs-material",
        "pymdown-extensions",
        "mkdocstrings",
        "mknotebooks",
        "pytkdocs_tweaks",
        "mkdocs_include_exclude_files",
        "jinja2"),
    make=() -> run(`mkdocs build`),
    target="site",
    devbranch="main")
