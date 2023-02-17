using Documenter, DocumenterMarkdown, LuxCore, Lux, LuxLib, Pkg

import Flux  # Load weak dependencies

function _setup_subdir_pkgs_index_file(subpkg)
    src_file = joinpath(dirname(@__DIR__), "lib", subpkg, "README.md")
    dst_file = joinpath(dirname(@__DIR__), "docs/src/lib", subpkg, "index.md")
    rm(dst_file; force=true)
    cp(src_file, dst_file)
    return
end

_setup_subdir_pkgs_index_file.(["Boltz", "LuxLib", "LuxCore"])

deployconfig = Documenter.auto_detect_deploy_system()
Documenter.post_status(deployconfig; type="pending", repo="github.com/avik-pal/Lux.jl.git")

makedocs(; sitename="Lux", authors="Avik Pal et al.", clean=true, doctest=true,
         modules=[Lux, LuxLib, LuxCore],
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
