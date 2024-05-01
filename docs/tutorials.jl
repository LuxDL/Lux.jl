using Distributed

#! format: off
BEGINNER_TUTORIALS = [
    "Basics/main.jl",
    "PolynomialFitting/main.jl",
    "SimpleRNN/main.jl",
    "SimpleChains/main.jl"
]
INTERMEDIATE_TUTORIALS = [
    "NeuralODE/main.jl",
    "BayesianNN/main.jl",
    "HyperNet/main.jl"
]
ADVANCED_TUTORIALS = [
    "GravitationalWaveForm/main.jl",
    "SymbolicOptimalControl/main.jl"
]

TUTORIALS = [
    collect(enumerate(Iterators.product(["beginner"], BEGINNER_TUTORIALS)))...,
    collect(enumerate(Iterators.product(["intermediate"], INTERMEDIATE_TUTORIALS)))...,
    collect(enumerate(Iterators.product(["advanced"], ADVANCED_TUTORIALS)))...
]
#! format: on

NWORKERS = min(parse(Int, get(ENV, "LUX_DOCUMENTATION_NWORKERS", "1")), length(TUTORIALS))

addprocs(NWORKERS;
    enable_threaded_blas=true,
    env=["JULIA_NUM_THREADS" => "$(Threads.nthreads())",
        "JULIA_CUDA_HARD_MEMORY_LIMIT" => "$(100 ÷ NWORKERS)",
        "JULIA_PKG_PRECOMPILE_AUTO" => "0", "JULIA_DEBUG" => "Literate"])

@info "Lux Tutorial Build Running tutorials with $(NWORKERS) workers."

@everywhere get_example_path(p) = joinpath(@__DIR__, "..", "examples", p)

@info "Starting tutorial build"

try
    pmap(TUTORIALS) do (i, (d, p))
        println("Running tutorial $(i): $(p) on worker $(myid())")
        path = get_example_path(p)
        name = "$(i)_$(first(rsplit(p, "/")))"
        output_directory = joinpath(joinpath(@__DIR__, "src", "tutorials"), d)
        tutorial_proj = dirname(path)
        file = joinpath(dirname(@__FILE__), "run_single_tutorial.jl")

        cmd = `$(Base.julia_cmd()) --color=yes --startup-file=no --project=$(tutorial_proj) "$(file)" "$(name)" "$(output_directory)" "$(path)"`
        @info "Running Command: $(cmd)"
        run(cmd)
    end
catch e
    rmprocs(workers()...)
    rethrow(e)
end
