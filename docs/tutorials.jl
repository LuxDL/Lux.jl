const BEGINNER_TUTORIALS = [
    "Basics/main.jl" => "CPU",
    "PolynomialFitting/main.jl" => "CUDA",
    "SimpleRNN/main.jl" => "CUDA",
    # Technically this is run on CPU but we need a better machine to run it
    "SimpleChains/main.jl" => "CUDA",
    "OptimizationIntegration/main.jl" => "CPU",
]
const INTERMEDIATE_TUTORIALS = [
    "NeuralODE/main.jl" => "CUDA",
    "BayesianNN/main.jl" => "CPU", # This is an empty tutorial, left to redirect to Turing
    "HyperNet/main.jl" => "CUDA",
    "PINN2DPDE/main.jl" => "CUDA",
    "ConvolutionalVAE/main.jl" => "CUDA",
    "GCN_Cora/main.jl" => "CUDA",
    "RealNVP/main.jl" => "CUDA",
]
const ADVANCED_TUTORIALS = ["GravitationalWaveForm/main.jl" => "CPU"]

const TUTORIALS = [
    collect(enumerate(Iterators.product(["beginner"], first.(BEGINNER_TUTORIALS))))...,
    collect(
        enumerate(Iterators.product(["intermediate"], first.(INTERMEDIATE_TUTORIALS)))
    )...,
    collect(enumerate(Iterators.product(["advanced"], first.(ADVANCED_TUTORIALS))))...,
]
const BACKEND_LIST =
    lowercase.([
        last.(BEGINNER_TUTORIALS)...,
        last.(INTERMEDIATE_TUTORIALS)...,
        last.(ADVANCED_TUTORIALS)...,
    ])

const BACKEND_GROUP = lowercase(get(ENV, "TUTORIAL_BACKEND_GROUP", "all"))

const BUILDKITE_PARALLEL_JOB_COUNT = parse(
    Int, get(ENV, "BUILDKITE_PARALLEL_JOB_COUNT", "-1")
)

const TUTORIALS_WITH_BACKEND = if BACKEND_GROUP == "all"
    TUTORIALS
else
    TUTORIALS[BACKEND_LIST .== BACKEND_GROUP]
end

const TUTORIALS_BUILDING = if BUILDKITE_PARALLEL_JOB_COUNT > 0
    id = parse(Int, ENV["BUILDKITE_PARALLEL_JOB"]) + 1 # Index starts from 0
    splits = Vector{Vector{eltype(TUTORIALS_WITH_BACKEND)}}(
        undef, BUILDKITE_PARALLEL_JOB_COUNT
    )
    for i in eachindex(TUTORIALS_WITH_BACKEND)
        idx = mod1(i, BUILDKITE_PARALLEL_JOB_COUNT)
        if !isassigned(splits, idx)
            splits[idx] = Vector{eltype(TUTORIALS_WITH_BACKEND)}()
        end
        push!(splits[idx], TUTORIALS_WITH_BACKEND[i])
    end
    (id > length(splits) || !isassigned(splits, id)) ? [] : splits[id]
else
    TUTORIALS_WITH_BACKEND
end

const NTASKS = min(
    parse(Int, get(ENV, "LUX_DOCUMENTATION_NTASKS", "1")), length(TUTORIALS_BUILDING)
)

@info "Building Tutorials:" TUTORIALS_BUILDING

@info "Starting Lux Tutorial Build with $(NTASKS) tasks."

run(
    `$(Base.julia_cmd()) --startup=no --code-coverage=user --threads=$(Threads.nthreads()) --project=@literate -e 'import Pkg; Pkg.add(["Literate", "InteractiveUtils"])'`,
)

asyncmap(TUTORIALS_BUILDING; ntasks=NTASKS) do (i, (d, p))
    @info "Running Tutorial $(i): $(p) on task $(current_task())"
    path = joinpath(@__DIR__, "..", "examples", p)
    name = "$(i)_$(first(rsplit(p, "/")))"
    output_directory = joinpath(@__DIR__, "src", "tutorials", d)
    tutorial_proj = dirname(path)
    file = joinpath(dirname(@__FILE__), "run_single_tutorial.jl")

    withenv(
        "JULIA_NUM_THREADS" => "$(Threads.nthreads())",
        "JULIA_CUDA_HARD_MEMORY_LIMIT" => "$(100 ÷ NTASKS)%",
        "JULIA_PKG_PRECOMPILE_AUTO" => "0",
        "JULIA_DEBUG" => "Literate",
    ) do
        run(
            `$(Base.julia_cmd()) --startup=no --code-coverage=user --threads=$(Threads.nthreads()) --project=$(tutorial_proj) "$(file)" "$(name)" "$(output_directory)" "$(path)"`,
        )
    end

    return nothing
end
