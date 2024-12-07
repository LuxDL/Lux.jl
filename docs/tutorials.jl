#! format: off
const BEGINNER_TUTORIALS = [
    "Basics/main.jl" => "CUDA",
    "PolynomialFitting/main.jl" => "CUDA",
    "SimpleRNN/main.jl" => "CUDA",
    "SimpleChains/main.jl" => "CUDA",
    "OptimizationIntegration/main.jl" => "CUDA",
]
const INTERMEDIATE_TUTORIALS = [
    "NeuralODE/main.jl" => "CUDA",
    "BayesianNN/main.jl" => "CPU",
    "HyperNet/main.jl" => "CUDA",
    "PINN2DPDE/main.jl" => "CUDA",
    "ConditionalVAE/main.jl" => "CUDA",
]
const ADVANCED_TUTORIALS = [
    "GravitationalWaveForm/main.jl" => "CPU",
]

const TUTORIALS = [
    collect(enumerate(Iterators.product(["beginner"], first.(BEGINNER_TUTORIALS))))...,
    collect(enumerate(Iterators.product(["intermediate"], first.(INTERMEDIATE_TUTORIALS))))...,
    collect(enumerate(Iterators.product(["advanced"], first.(ADVANCED_TUTORIALS))))...
]
const BACKEND_LIST = lowercase.([
    last.(BEGINNER_TUTORIALS)...,
    last.(INTERMEDIATE_TUTORIALS)...,
    last.(ADVANCED_TUTORIALS)...
])
#! format: on

const BACKEND_GROUP = lowercase(get(ENV, "TUTORIAL_BACKEND_GROUP", "all"))

const BUILDKITE_PARALLEL_JOB_COUNT = parse(
    Int, get(ENV, "BUILDKITE_PARALLEL_JOB_COUNT", "-1"))

const TUTORIALS_WITH_BACKEND = if BACKEND_GROUP == "all"
    TUTORIALS
else
    TUTORIALS[BACKEND_LIST .== BACKEND_GROUP]
end

const TUTORIALS_BUILDING = if BUILDKITE_PARALLEL_JOB_COUNT > 0
    id = parse(Int, ENV["BUILDKITE_PARALLEL_JOB"]) + 1 # Index starts from 0
    splits = collect(Iterators.partition(TUTORIALS_WITH_BACKEND,
        cld(length(TUTORIALS_WITH_BACKEND), BUILDKITE_PARALLEL_JOB_COUNT)))
    id > length(splits) ? [] : splits[id]
else
    TUTORIALS_WITH_BACKEND
end

const NTASKS = min(
    parse(Int, get(ENV, "LUX_DOCUMENTATION_NTASKS", "1")), length(TUTORIALS_BUILDING))

@info "Building Tutorials:" TUTORIALS_BUILDING

@info "Starting Lux Tutorial Build with $(NTASKS) tasks."

run(`$(Base.julia_cmd()) --startup=no --code-coverage=user --threads=$(Threads.nthreads()) --project=@literate -e 'import Pkg; Pkg.add(["Literate", "InteractiveUtils"])'`)

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
        run(`$(Base.julia_cmd()) --startup=no --code-coverage=user --threads=$(Threads.nthreads()) --project=$(tutorial_proj) "$(file)" "$(name)" "$(output_directory)" "$(path)"`)
    end

    return
end
