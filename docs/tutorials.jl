#! format: off
const BEGINNER_TUTORIALS = [
    # tutorial name             device   should_run?
    ("Basics",                  "CPU",   true),
    ("PolynomialFitting",       "CUDA",  true),
    ("SimpleRNN",               "CUDA",  true),
    ("SimpleChains",            "CUDA",  true),
    ("OptimizationIntegration", "CPU",   true),
]
const INTERMEDIATE_TUTORIALS = [
    ("NeuralODE",               "CUDA", false),
    ("BayesianNN",              "CPU",  false),
    ("HyperNet",                "CUDA", true),
    ("PINN2DPDE",               "CUDA", true),
    ("ConvolutionalVAE",        "CUDA", true),
    ("GCN_Cora",                "CUDA", true),
    ("RealNVP",                 "CPU",  false),
    ("LSTMEncoderDecoder",      "CPU",  true),
]
const ADVANCED_TUTORIALS = [
    ("GravitationalWaveForm",   "CPU",  true),
]
#! format: on

get_name_and_run(list) = getindex.(list, ([1, 3],))

const TUTORIALS = [
    collect(
        enumerate(Iterators.product(["beginner"], get_name_and_run(BEGINNER_TUTORIALS)))
    )...,
    collect(
        enumerate(
            Iterators.product(["intermediate"], get_name_and_run(INTERMEDIATE_TUTORIALS))
        ),
    )...,
    collect(
        enumerate(Iterators.product(["advanced"], get_name_and_run(ADVANCED_TUTORIALS)))
    )...,
]
const BACKEND_LIST =
    lowercase.([
        getindex.(BEGINNER_TUTORIALS, 2)...,
        getindex.(INTERMEDIATE_TUTORIALS, 2)...,
        getindex.(ADVANCED_TUTORIALS, 2)...,
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

asyncmap(TUTORIALS_BUILDING; ntasks=NTASKS) do (i, (d, (p, should_run)))
    @info "Running Tutorial $(i): $(p) on task $(current_task())"
    path = joinpath(@__DIR__, "..", "examples", p, "main.jl")
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
            `$(Base.julia_cmd()) --startup=no --code-coverage=user --threads=$(Threads.nthreads()) --project=$(tutorial_proj) "$(file)" "$(name)" "$(output_directory)" "$(path)" "$(should_run)"`,
        )
    end

    return nothing
end
