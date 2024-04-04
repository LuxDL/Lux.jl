using BenchmarkTools: BenchmarkTools, BenchmarkGroup, @btime, @benchmarkable
using ComponentArrays: ComponentArray
using InteractiveUtils: versioninfo
using Lux: Lux, BatchNorm, Chain, Conv, Dense, Dropout, FlattenLayer, MaxPool
using NNlib: relu
using StableRNGs: StableRNG
using Statistics: median

@info sprint(versioninfo)

const SUITE = BenchmarkGroup()

include("helpers.jl")
include("vgg.jl")
include("layers.jl")

BenchmarkTools.tune!(SUITE)
results = BenchmarkTools.run(SUITE; verbose=true)

BenchmarkTools.save(joinpath(@__DIR__, "benchmark_results.json"), median(results))
