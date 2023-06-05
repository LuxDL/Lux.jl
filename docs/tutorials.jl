using Literate

get_example_path(p) = joinpath(@__DIR__, "..", "examples", p)
OUTPUT = joinpath(@__DIR__, "src", "examples", "generated")

BEGINNER_TUTORIALS = ["Basics/main.jl", "PolynomialFitting/main.jl", "SimpleRNN/main.jl"]
INTERMEDIATE_TUTORIALS = ["NeuralODE/main.jl", "BayesianNN/main.jl", "HyperNet/main.jl"]
ADVANCED_TUTORIALS = []

for (d, paths) in (("beginner", BEGINNER_TUTORIALS),
    ("intermediate", INTERMEDIATE_TUTORIALS),
    ("advanced", ADVANCED_TUTORIALS))
    for p in paths
        Literate.markdown(get_example_path(p),
            joinpath(OUTPUT, d, dirname(p));
            documenter=true)
    end
end
