using ADTypes: ADTypes, AutoEnzyme, AutoTracker, AutoReverseDiff, AutoTapir, AutoZygote
using BenchmarkTools: BenchmarkTools, BenchmarkGroup, @btime, @benchmarkable
using ComponentArrays: ComponentArray
using InteractiveUtils: versioninfo
using FastClosures: @closure
using Flux: Flux
using Functors: fmap
using LinearAlgebra: BLAS
using Lux: Lux, BatchNorm, Chain, Conv, Dense, Dropout, FlattenLayer, MaxPool
using NNlib: relu
using SimpleChains: SimpleChains, static
using StableRNGs: StableRNG
using Statistics: median

# AD Backends
using Enzyme: Enzyme
using ReverseDiff: ReverseDiff
using Tapir: Tapir
using Tracker: Tracker
using Zygote: Zygote

BLAS.set_num_threads(min(4, Threads.nthreads()))

@info sprint(versioninfo)
@info "BLAS threads: $(BLAS.get_num_threads())"

const SUITE = BenchmarkGroup()

include("helpers.jl")
include("vgg.jl")
include("layers.jl")

BenchmarkTools.tune!(SUITE; verbose=true)
results = BenchmarkTools.run(SUITE; verbose=true)
display(median(results))

BenchmarkTools.save(joinpath(@__DIR__, "benchmark_results.json"), median(results))
