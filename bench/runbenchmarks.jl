using ADTypes: ADTypes, AutoEnzyme, AutoTracker, AutoReverseDiff, AutoZygote
using BenchmarkTools: BenchmarkTools, BenchmarkGroup, @btime, @benchmarkable
using ComponentArrays: ComponentArray
using InteractiveUtils: versioninfo
using FastClosures: @closure
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

# BenchmarkTools Parameters
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.25

struct AutoTapir <: ADTypes.AbstractReverseMode end

@info sprint(versioninfo)

const SUITE = BenchmarkGroup()

include("helpers.jl")
include("vgg.jl")
include("layers.jl")

BenchmarkTools.tune!(SUITE; verbose=true)
results = BenchmarkTools.run(SUITE; verbose=true)

BenchmarkTools.save(joinpath(@__DIR__, "benchmark_results.json"), median(results))
