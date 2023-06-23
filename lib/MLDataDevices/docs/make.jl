using Documenter, DocumenterMarkdown, LuxDeviceUtils

deployconfig = Documenter.auto_detect_deploy_system()
Documenter.post_status(deployconfig;
    type="pending",
    repo="github.com/LuxDL/LuxDeviceUtils.jl.git")

makedocs(;
    sitename="LuxDeviceUtils",
    authors="Avik Pal et al.",
    clean=true,
    doctest=true,
    modules=[LuxDeviceUtils],
    strict=[:doctest, :linkcheck, :parse_error, :example_block, :missing_docs],
    checkdocs=:all,
    format=Markdown(),
    draft=false,
    build=joinpath(@__DIR__, "docs"))

deploydocs(;
    repo="github.com/LuxDL/LuxDeviceUtils.jl.git",
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
