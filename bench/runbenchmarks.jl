using BenchmarkTools: BenchmarkTools, BenchmarkGroup, @btime, @benchmarkable
using ComponentArrays: ComponentArray
using Lux: Lux, BatchNorm, Chain, Conv, Dense, Dropout, FlattenLayer, MaxPool
using NNlib: relu
using StableRNGs: StableRNG
using Statistics: median

const SUITE = BenchmarkGroup()

include("helpers.jl")
include("vgg.jl")

BenchmarkTools.tune!(SUITE)
results = BenchmarkTools.run(SUITE; verbose=true)

BenchmarkTools.save(joinpath(@__DIR__, "benchmark_results.json"), median(results))
