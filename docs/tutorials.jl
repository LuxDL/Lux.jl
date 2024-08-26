#! format: off
const BEGINNER_TUTORIALS = [
    "Basics/main.jl",
    "PolynomialFitting/main.jl",
    "SimpleRNN/main.jl",
    "SimpleChains/main.jl"
]
const INTERMEDIATE_TUTORIALS = [
    "NeuralODE/main.jl",
    "BayesianNN/main.jl",
    "HyperNet/main.jl"
]
const ADVANCED_TUTORIALS = [
    "GravitationalWaveForm/main.jl",
    "SymbolicOptimalControl/main.jl"
]

const TUTORIALS = [
    collect(enumerate(Iterators.product(["beginner"], BEGINNER_TUTORIALS)))...,
    collect(enumerate(Iterators.product(["intermediate"], INTERMEDIATE_TUTORIALS)))...,
    collect(enumerate(Iterators.product(["advanced"], ADVANCED_TUTORIALS)))...
]
#! format: on

const BUILDKITE_PARALLEL_JOB_COUNT = parse(
    Int, get(ENV, "BUILDKITE_PARALLEL_JOB_COUNT", "-1"))

const TUTORIALS_BUILDING = if BUILDKITE_PARALLEL_JOB_COUNT > 0
    id = parse(Int, ENV["BUILDKITE_PARALLEL_JOB"]) + 1 # Index starts from 0
    splits = collect(Iterators.partition(
        TUTORIALS, cld(length(TUTORIALS), BUILDKITE_PARALLEL_JOB_COUNT)))
    id > length(splits) ? [] : splits[id]
else
    TUTORIALS
end

const NTASKS = min(
    parse(Int, get(ENV, "LUX_DOCUMENTATION_NTASKS", "1")), length(TUTORIALS_BUILDING))

@info "Building Tutorials:" TUTORIALS_BUILDING

@info "Starting Lux Tutorial Build with $(NTASKS) tasks."

asyncmap(TUTORIALS_BUILDING; ntasks=NTASKS) do (i, (d, p))
    @info "Running Tutorial $(i): $(p) on task $(current_task())"
    path = joinpath(@__DIR__, "..", "examples", p)
    name = "$(i)_$(first(rsplit(p, "/")))"
    output_directory = joinpath(@__DIR__, "src", "tutorials", d)
    tutorial_proj = dirname(path)
    file = joinpath(dirname(@__FILE__), "run_single_tutorial.jl")

    withenv("JULIA_NUM_THREADS" => "$(Threads.nthreads())",
        "JULIA_CUDA_HARD_MEMORY_LIMIT" => "$(100 ÷ NTASKS)%",
        "JULIA_PKG_PRECOMPILE_AUTO" => "0", "JULIA_DEBUG" => "Literate") do
        cmd = `$(Base.julia_cmd()) --code-coverage=user --threads=$(Threads.nthreads()) --project=$(tutorial_proj) "$(file)" "$(name)" "$(output_directory)" "$(path)"`
        run(cmd)
    end

    return
end
