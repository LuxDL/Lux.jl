using BenchmarkTools: BenchmarkTools, BenchmarkGroup, Trial, @btime, @benchmarkable
using Dates: Dates
using InteractiveUtils: versioninfo
using FastClosures: @closure
using Functors: fmap
using Git: git
using JSON3: JSON3
using LinearAlgebra: BLAS
using Logging: Logging
using StableRNGs: StableRNG
using Statistics: median, mean
using ThreadPinning: ThreadPinning

# NN specific
using ComponentArrays: ComponentArray
using Flux: Flux
using Lux: Lux, BatchNorm, Chain, Conv, Dense, Dropout, FlattenLayer, MaxPool
using LuxCUDA: LuxCUDA, CUDA
using LuxDeviceUtils: LuxCPUDevice, LuxCUDADevice, get_device
using NNlib: relu
using SimpleChains: SimpleChains, static

# AD Backends
using ADTypes: ADTypes, AutoEnzyme, AutoTapir, AutoTracker, AutoReverseDiff, AutoZygote
using Enzyme: Enzyme
using ReverseDiff: ReverseDiff
using Tapir: Tapir
using Tracker: Tracker
using Zygote: Zygote

const LUX_TRIAL_BENCHMARK = parse(Bool, get(ENV, "LUX_TRIAL_BENCHMARK", "false")) # Don't submit results if true

const GITHUB_TOKEN = get(ENV, "GITHUB_TOKEN", nothing)

const DO_GITHUB_STUFF = GITHUB_TOKEN !== nothing

if !LUX_TRIAL_BENCHMARK && !DO_GITHUB_STUFF
    error("`LUX_TRIAL_BENCHMARK` is false, but `GITHUB_TOKEN` is not set.")
end

const github_directory = mktempdir()
const cur_directory = pwd()

include("generate_html_report.jl")
include("git.jl")

function do_github_stuff(f::F, dir=nothing) where {F}
    if DO_GITHUB_STUFF
        cd(github_directory)
        dir === nothing || cd(dir)
        @info "Running github stuff in $(pwd())"
        f()
        cd(cur_directory)
    end
end

do_github_stuff() do
    run(`$(git()) clone https://www.github.com/LuxDL/Lux.jl.git`)
    cd("Lux.jl")
    # Set remote url for git
    run(`$(git()) remote set-url origin https://avik-pal:$(GITHUB_TOKEN)@github.com/LuxDL/Lux.jl.git`)
    run(`$(git()) fetch origin`)
end

if readchomp(`$(git()) rev-parse --abbrev-ref HEAD`) == "main" && LUX_TRIAL_BENCHMARK
    @warn "The current branch is main but `LUX_TRIAL_BENCHMARK` is set to true."
end

BLAS.set_num_threads(Threads.nthreads() ÷ 2)
ThreadPinning.pinthreads(:cores)
ThreadPinning.threadinfo()

@info sprint(versioninfo)
@info sprint(CUDA.versioninfo)
@info "BLAS threads: $(BLAS.get_num_threads())"

const SUITE = BenchmarkGroup()

include("helpers.jl")
include("vgg.jl")
include("layers.jl")

params_file = joinpath(@__DIR__, "params.json")
remote_params_file = joinpath(github_directory, "Lux.jl", "params.json")

const cur_branch = readchomp(`$(git()) rev-parse --abbrev-ref HEAD`)

@info "Current branch: $(cur_branch)"

do_github_stuff("Lux.jl") do
    run(`$(git()) checkout origin/benchmark-results`)
    run(`$(git()) switch -c origin/benchmark-results`)
    run(`$(git()) pull origin benchmark-results`)
    if isfile(remote_params_file)
        cp(remote_params_file, params_file)
        @info "Copied params file from remote."
    else
        @warn "No params file found. This is probably the first time running benchmark."
    end
end

if isfile(params_file)
    loaded_params = BenchmarkTools.load(params_file)[1]
    tune_or_load!(SUITE, loaded_params) # Load only if possible
else
    BenchmarkTools.tune!(SUITE; verbose=true)
end
BenchmarkTools.save(params_file, BenchmarkTools.params(SUITE))

do_github_stuff("Lux.jl") do
    cp(params_file, remote_params_file)
    run(`$(git()) add $(remote_params_file)`)
    run(`$(git()) commit -m "Add updated benchmark parameters"`)
    run(`$(git()) push heads/origin/benchmark-results`)
end

results = BenchmarkTools.run(SUITE; verbose=true)

old_benchmarks = nothing
old_benchmarks_path = joinpath(github_directory, "Lux.jl", "last_benchmark_results.json")

do_github_stuff() do
    if isfile(old_benchmarks_path)
        global old_benchmarks = BenchmarkTools.load(old_benchmarks_path)
    else
        @warn "No old benchmark results found. This is probably the first time running benchmark."
    end
end

# TODO: We store the result of last benchmark to a file, and use BenchmarkTools.judge and
# post the results to the PR. This is a bit more involved, and we will do it later.

current_benchmarks = joinpath(@__DIR__, "last_benchmark_results.json")
BenchmarkTools.save(current_benchmarks, median(results))

do_github_stuff() do
    cp(current_benchmarks, remote_benchmarks)
    run(`$(git()) add $(remote_benchmarks)`)
    run(`$(git()) commit -m "Add updated benchmark results"`)
    run(`$(git()) push heads/origin/benchmark-results`)
end

flat_results = flatten_benchmark(results)
flat_results_json = JSON3.write(flat_results)

fname = joinpath(github_directory, "Lux.jl", "benchmark_results.json")
final_results = isfile(fname) ? [JSON3.read(open(fname, "r"))..., flat_results] :
                [flat_results]

open(fname, "w") do f
    JSON3.pretty(f, JSON3.write(final_results))
    println(f)
end

@info "Benchmark results written to $(fname)"

do_github_stuff() do
    run(`$(git()) add $(fname)`)
    run(`$(git()) commit -m "Add updated benchmark results"`)
    run(`$(git()) push heads/origin/benchmark-results`)
end
