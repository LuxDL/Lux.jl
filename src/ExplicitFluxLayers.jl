module ExplicitFluxLayers

using Statistics, NNlib, CUDA, Random, Setfield, ChainRulesCore
import NNlibCUDA: batchnorm
import Flux
import Flux:
    zeros32,
    ones32,
    glorot_normal,
    glorot_uniform,
    convfilter,
    expand,
    calc_padding,
    DenseConvDims,
    _maybetuple_string,
    reshape_cell_output

# Base Type
abstract type AbstractExplicitLayer end

initialparameters(::AbstractRNG, ::Any) = NamedTuple()
initialparameters(l) = initialparameters(Random.GLOBAL_RNG, l)
initialstates(::AbstractRNG, ::Any) = NamedTuple()
initialstates(l) = initialstates(Random.GLOBAL_RNG, l)

function initialparameters(rng::AbstractRNG, l::NamedTuple)
    return NamedTuple{Tuple(collect(keys(l)))}(initialparameters.((rng,), values(l)))
end
initialstates(rng::AbstractRNG, l::NamedTuple) = NamedTuple{Tuple(collect(keys(l)))}(initialstates.((rng,), values(l)))

setup(rng::AbstractRNG, l::AbstractExplicitLayer) = (initialparameters(rng, l), initialstates(rng, l))
setup(l::AbstractExplicitLayer) = setup(Random.GLOBAL_RNG, l)

nestedtupleofarrayslength(t::Any) = 1
nestedtupleofarrayslength(t::AbstractArray) = length(t)
function nestedtupleofarrayslength(t::Union{NamedTuple,Tuple})
    length(t) == 0 && return 0
    return sum(nestedtupleofarrayslength, t)
end

parameterlength(l::AbstractExplicitLayer) = parameterlength(initialparameters(l))
statelength(l::AbstractExplicitLayer) = statelength(initialstates(l))
parameterlength(ps::NamedTuple) = nestedtupleofarrayslength(ps)
statelength(st::NamedTuple) = nestedtupleofarrayslength(st)

apply(model::AbstractExplicitLayer, x, ps::NamedTuple, s::NamedTuple) = model(x, ps, s)

# Test Mode
function testmode(states::NamedTuple, mode::Bool=true)
    updated_states = []
    for (k, v) in pairs(states)
        if k == :training
            push!(updated_states, k => !mode)
            continue
        end
        push!(updated_states, k => testmode(v, mode))
    end
    return (; updated_states...)
end

testmode(x::Any, mode::Bool=true) = x

testmode(m::AbstractExplicitLayer, mode::Bool=true) = testmode(initialstates(m), mode)

trainmode(x::Any, mode::Bool=true) = testmode(x, !mode)

# Utilities
zeros32(rng::AbstractRNG, args...; kwargs...) = zeros32(args...; kwargs...)
ones32(rng::AbstractRNG, args...; kwargs...) = ones32(args...; kwargs...)
Base.zeros(rng::AbstractRNG, args...; kwargs...) = zeros(args...; kwargs...)
Base.ones(rng::AbstractRNG, args...; kwargs...) = ones(args...; kwargs...)

include("norm_utils.jl")

# Layer Implementations
include("chain.jl")
include("batchnorm.jl")
include("linear.jl")
include("convolution.jl")
include("pooling.jl")
include("weightnorm.jl")
include("basics.jl")

# Transition to Explicit Layers
include("transform.jl")

# Pretty Printing
include("show_layers.jl")

end
