using Documenter, Lux, Literate, Pkg

# Precompile example dependencies
Pkg.activate(joinpath(@__DIR__, "..", "examples"))
Pkg.develop(PackageSpec(; path=joinpath(@__DIR__, "..")))
Pkg.instantiate()
Pkg.precompile()
Pkg.activate(@__DIR__)

if haskey(ENV, "GITHUB_ACTIONS")
    ENV["JULIA_DEBUG"] = "Documenter"
    ENV["DATADEPS_ALWAYS_ACCEPT"] = true
end

deployconfig = Documenter.auto_detect_deploy_system()
Documenter.post_status(deployconfig; type="pending", repo="github.com/avik-pal/Lux.jl.git")

# Tutorials
get_example_path(p) = joinpath(@__DIR__, "..", "examples", p)
OUTPUT = joinpath(@__DIR__, "src", "examples", "generated")

BEGINNER_TUTORIALS = []
INTERMEDIATE_TUTORIALS = ["NeuralODE/main.jl"]
ADVANCED_TUTORIALS = []
MAPPING = Dict("beginner" => [], "intermediate" => [], "advanced" => [])

for (d, paths) in
    (("beginner", BEGINNER_TUTORIALS), ("intermediate", INTERMEDIATE_TUTORIALS), ("advanced", ADVANCED_TUTORIALS))
    for p in paths
        Literate.markdown(get_example_path(p), joinpath(OUTPUT, d, dirname(p)); documenter=true)
        push!(MAPPING[d], dirname(p) => joinpath("examples/generated", d, dirname(p), splitext(basename(p))[1] * ".md"))
    end
end

display(MAPPING)

makedocs(;
    sitename="Lux",
    authors="Avik Pal et al.",
    clean=true,
    doctest=false,
    modules=[Lux],
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        assets=["assets/custom.css"],
        # analytics = ""
    ),
    pages=[
        "Lux: Explicitly Parameterized Neural Networks" => "index.md",
        "Introduction" => ["All about Lux" => "introduction/overview.md", "Ecosystem" => "introduction/ecosystem.md"],
        "Examples" => [
            "Beginner" => MAPPING["beginner"],
            "Intermediate" => MAPPING["intermediate"],
            "Advanced" => MAPPING["advanced"],
            "Additional Examples" => "examples.md",
        ],
        "API" => [
            "Layers" => "api/layers.md",
            "Functional" => "api/functional.md",
            "Core" => "api/core.md",
            "Utilities" => "api/utilities.md",
        ],
    ],
)

deploydocs(; repo="github.com/avik-pal/Lux.jl.git", push_preview=true, devbranch="main")

Pkg.activate(@__DIR__)
