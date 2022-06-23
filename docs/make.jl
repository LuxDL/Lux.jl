using Documenter, DocumenterMarkdown, Lux, Literate, Pkg

# Precompile example dependencies
Pkg.activate(joinpath(@__DIR__, "..", "examples"))
Pkg.develop(PackageSpec(; path=joinpath(@__DIR__, "..")))
Pkg.instantiate()
Pkg.precompile()
Pkg.activate(@__DIR__)

using DiffEqSensitivity

if haskey(ENV, "GITHUB_ACTIONS")
    ENV["JULIA_DEBUG"] = "Documenter"
    ENV["DATADEPS_ALWAYS_ACCEPT"] = true
end

deployconfig = Documenter.auto_detect_deploy_system()
Documenter.post_status(deployconfig; type="pending",
                       repo="github.com/avik-pal/Lux.jl.git")

# Tutorials
get_example_path(p) = joinpath(@__DIR__, "..", "examples", p)
OUTPUT = joinpath(@__DIR__, "src", "examples", "generated")

BEGINNER_TUTORIALS = ["Basics/main.jl", "SimpleRNN/main.jl"]
BEGINNER_TUTORIAL_NAMES = ["Julia & Lux for the Uninitiated", "Training a Simple LSTM"]
INTERMEDIATE_TUTORIALS = ["NeuralODE/main.jl", "BayesianNN/main.jl"]
INTERMEDIATE_TUTORIAL_NAMES = ["MNIST NeuralODE Classification", "Bayesian Neural Network"]
ADVANCED_TUTORIALS = []
ADVANCED_TUTORIAL_NAMES = []
MAPPING = Dict("beginner" => [], "intermediate" => [], "advanced" => [])

for (d, names, paths) in (("beginner", BEGINNER_TUTORIAL_NAMES, BEGINNER_TUTORIALS),
                          ("intermediate", INTERMEDIATE_TUTORIAL_NAMES,
                           INTERMEDIATE_TUTORIALS),
                          ("advanced", ADVANCED_TUTORIAL_NAMES, ADVANCED_TUTORIALS))
    for (n, p) in zip(names, paths)
        Literate.markdown(get_example_path(p), joinpath(OUTPUT, d, dirname(p));
                          documenter=true)
        push!(MAPPING[d],
              n => joinpath("examples/generated", d, dirname(p),
                            splitext(basename(p))[1] * ".md"))
    end
end

display(MAPPING)

makedocs(;
         sitename="Lux",
         authors="Avik Pal et al.",
         clean=true,
         doctest=true,
         modules=[Lux],
         strict=[
             :doctest,
             :linkcheck,
             :parse_error,
             :example_block,
             # Other available options are
             # :autodocs_block, :cross_references, :docs_block, :eval_block, :example_block,
             # :footnote, :meta_block, :missing_docs, :setup_block
         ],
         checkdocs=:all,
         format=Markdown(),
         draft=false,
         build=joinpath(@__DIR__, "docs"))

deploydocs(; repo="github.com/avik-pal/Lux.jl.git", push_preview=true,
           deps=Deps.pip("mkdocs", "pygments", "python-markdown-math",
                         "mkdocs-material", "pymdown-extensions", "mkdocstrings",
                         "mknotebooks", "pytkdocs_tweaks", "mkdocs_include_exclude_files",
                         "jinja2"),
           make=() -> run(`mkdocs build`), target="site", devbranch="main")

Pkg.activate(@__DIR__)
