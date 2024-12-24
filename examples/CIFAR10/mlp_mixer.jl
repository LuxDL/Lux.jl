using Comonicon, Lux, Optimisers, Printf, Random, Statistics, Zygote, Enzyme

CUDA.allowscalar(false)

@isdefined(includet) ? includet("common.jl") : include("common.jl")

