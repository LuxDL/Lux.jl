using Distributed

addprocs(parse(Int, get(ENV, "LUX_DOCUMENTATION_NWORKERS", "1")))

@everywhere const LUX_DOCUMENTATION_NWORKERS = parse(
    Int, get(ENV, "LUX_DOCUMENTATION_NWORKERS", "1"))
@info "Lux Tutorial Build Running tutorials with $(LUX_DOCUMENTATION_NWORKERS) workers."
@everywhere const CUDA_MEMORY_LIMIT = 100 ÷ LUX_DOCUMENTATION_NWORKERS

@everywhere using Literate

@everywhere function preprocess(path, str)
    return replace(str, "__DIR = @__DIR__" => "__DIR = \"$(dirname(path))\"")
end

@everywhere get_example_path(p) = joinpath(@__DIR__, "..", "examples", p)

BEGINNER_TUTORIALS = ["Basics/main.jl", "PolynomialFitting/main.jl", "SimpleRNN/main.jl"]
INTERMEDIATE_TUTORIALS = ["NeuralODE/main.jl", "BayesianNN/main.jl", "HyperNet/main.jl"]
ADVANCED_TUTORIALS = ["GravitationalWaveForm/main.jl"]

TUTORIALS = [collect(Iterators.product(["beginner"], BEGINNER_TUTORIALS))...,
    collect(Iterators.product(["intermediate"], INTERMEDIATE_TUTORIALS))...,
    collect(Iterators.product(["advanced"], ADVANCED_TUTORIALS))...]

pmap(enumerate(TUTORIALS)) do (i, (d, p))
    println("Running tutorial $(i): $(p) on worker $(myid())")
    withenv("JULIA_DEBUG" => "Literate",
        "JULIA_CUDA_SOFT_MEMORY_LIMIT" => "$(CUDA_MEMORY_LIMIT)%") do
        name = "$(i)_$(first(rsplit(p, "/")))"
        p_ = get_example_path(p)
        OUTPUT = joinpath(@__DIR__, "src", "tutorials")
        return Literate.markdown(
            p_, joinpath(OUTPUT, d); execute=true, name, documenter=true,
            preprocess=Base.Fix1(preprocess, p_))
    end
end
