using Literate

get_example_path(p) = joinpath(@__DIR__, "..", "examples", p)
OUTPUT = joinpath(@__DIR__, "src", "tutorials")

function preprocess(path, str)
    dir = dirname(path)
    str = replace(str, "__DIR = @__DIR__" => "__DIR = \"$dir\"")
    return str
end

BEGINNER_TUTORIALS = ["Basics/main.jl", "PolynomialFitting/main.jl", "SimpleRNN/main.jl"]
INTERMEDIATE_TUTORIALS = ["NeuralODE/main.jl", "BayesianNN/main.jl", "HyperNet/main.jl"]
ADVANCED_TUTORIALS = ["GravitationalWaveForm/main.jl"]

for (d, paths) in (("beginner", BEGINNER_TUTORIALS),
    ("intermediate", INTERMEDIATE_TUTORIALS), ("advanced", ADVANCED_TUTORIALS))
    for (i, p) in enumerate(paths)
        name = "$(i)_$(first(rsplit(p, "/")))"
        p_ = get_example_path(p)
        Literate.markdown(p_, joinpath(OUTPUT, d); documenter=true, name,
            preprocess=Base.Fix1(preprocess, p_))
    end
end
