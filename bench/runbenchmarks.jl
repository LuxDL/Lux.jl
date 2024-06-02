using BenchmarkTools: BenchmarkTools, BenchmarkGroup, Trial, @btime, @benchmarkable
using Dates: Dates
using InteractiveUtils: versioninfo
using FastClosures: @closure
using Functors: fmap
using Git: git
using LinearAlgebra: BLAS
using StableRNGs: StableRNG
using Statistics: median
using ThreadPinning: ThreadPinning

# NN specific
using ComponentArrays: ComponentArray
using Flux: Flux
using Lux: Lux, BatchNorm, Chain, Conv, Dense, Dropout, FlattenLayer, MaxPool
using LuxCUDA: LuxCUDA, CUDA
using LuxDeviceUtils: LuxCPUDevice, LuxCUDADevice
using NNlib: relu
using SimpleChains: SimpleChains, static

# AD Backends
using ADTypes: ADTypes, AutoEnzyme, AutoTapir, AutoTracker, AutoReverseDiff, AutoZygote
using Enzyme: Enzyme
using ReverseDiff: ReverseDiff
using Tapir: Tapir
using Tracker: Tracker
using Zygote: Zygote

BLAS.set_num_threads(min(4, Threads.nthreads()))
ThreadPinning.pinthreads(:cores)
ThreadPinning.threadinfo()

@info sprint(versioninfo)
@info "BLAS threads: $(BLAS.get_num_threads())"

const SUITE = BenchmarkGroup()

include("helpers.jl")
include("vgg.jl")
include("layers.jl")

BenchmarkTools.tune!(SUITE; verbose=true)
results = BenchmarkTools.run(SUITE; verbose=true)

flat_results = flatten_benchmark(results)
JSON.print(flat_results, 4)

BenchmarkTools.save(joinpath(@__DIR__, "benchmark_results.json"), median(results))
