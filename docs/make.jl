using Documenter, DocumenterMarkdown, Flux2Lux, Lux, LuxLib, Pkg

deployconfig = Documenter.auto_detect_deploy_system()
Documenter.post_status(deployconfig; type="pending", repo="github.com/avik-pal/Lux.jl.git")

makedocs(; sitename="Lux", authors="Avik Pal et al.", clean=true, doctest=true,
         modules=[Flux2Lux, Lux, LuxLib],
         strict=[
             :doctest,
             :linkcheck,
             :parse_error,
             :example_block,
             # Other available options are
             # :autodocs_block, :cross_references, :docs_block, :eval_block, :example_block,
             # :footnote, :meta_block, :missing_docs, :setup_block
         ], checkdocs=:all, format=Markdown(), draft=false,
         build=joinpath(@__DIR__, "docs"))

deploydocs(; repo="github.com/avik-pal/Lux.jl.git", push_preview=true,
           deps=Deps.pip("mkdocs", "pygments", "python-markdown-math", "mkdocs-material",
                         "pymdown-extensions", "mkdocstrings", "mknotebooks",
                         "pytkdocs_tweaks", "mkdocs_include_exclude_files", "jinja2"),
           make=() -> run(`mkdocs build`), target="site", devbranch="main")
