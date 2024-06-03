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

# Plotting and Pushing data
using PlotlyJS: PlotlyJS

const TRIAL_BENCHMARK = get(ENV, "TRIAL_BENCHMARK", false) # Don't submit results if true

if readchomp(`$(git()) rev-parse --abbrev-ref HEAD`) == "main" && TRIAL_BENCHMARK
    @warn "The current branch is main but `TRIAL_BENCHMARK` is set to true."
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

tuning_finished = false
if isfile(params_file)
    # Check if the benchmarks haven't changed
    try
        BenchmarkTools.loadparams!(SUITE, BenchmarkTools.load(params_file)[1], :evals, :samples)
        global tuning_finished = true
    catch err
        @error err
        @warn "Benchmarks have changed, re-tuning."
    end
end

if !tuning_finished
    BenchmarkTools.tune!(SUITE; verbose=true)
    BenchmarkTools.save(params_file, BenchmarkTools.params(SUITE))
end

results = BenchmarkTools.run(SUITE; verbose=true)

flat_results = flatten_benchmark(results)
flat_results_json = JSON3.write(flat_results)

fname = joinpath(@__DIR__, "benchmark_results.json")

final_results = isfile(fname) ? [JSON3.read(open(fname, "r"))..., flat_results] : [flat_results]

open(fname, "w") do f
    JSON3.pretty(f, JSON3.write(final_results))
    println(f)
end

@info "Benchmark results written to $(fname)"

# TODO: We store the result of last benchmark to a file, and use BenchmarkTools.judge

# Now we cleanup the benchmarks, and generate nice looking plots

include("prettify.jl")
