"""
    Embedding(in_dims => out_dims; init_weight=rand32)

A lookup table that stores embeddings of dimension `out_dims` for a vocabulary of size
`in_dims`. When the vocabulary is multi-dimensional, the input is expected to be a tuple
of Cartesian indices.

This layer is often used to store word embeddings and retrieve them using indices.

## Arguments

  - `in_dims`: number(s) of input dimensions
  - `out_dims`: number of output dimensions

## Keyword Arguments

  - `init_weight`: initializer for the weight matrix
    (`weight = init_weight(rng, out_dims, in_dims...)`)

## Input

  - Integer OR
  - Abstract Vector of Integers OR
  - Abstract Array of Integers OR
  - Tuple of Integers OR
  - Tuple of Abstract Vectors of Integers OR
  - Tuple of Abstract Arrays of Integers

## Returns

  - Returns the embedding corresponding to each index in the input. For an N dimensional
    input, an N + 1 dimensional output is returned.
  - Empty `NamedTuple()`
"""
@concrete struct Embedding <: AbstractLuxLayer
    in_dims <: Union{IntegerType, Tuple{Vararg{IntegerType}}}
    out_dims <: IntegerType
    init_weight
end

function Embedding((in_dims, out_dims)::Pair; init_weight=rand32)
    return Embedding(in_dims, out_dims, init_weight)
end

function initialparameters(rng::AbstractRNG, e::Embedding)
    return (weight=e.init_weight(rng, e.out_dims, e.in_dims...),)
end

function Base.show(io::IO, e::Embedding)
    return print(io, "Embedding(", e.in_dims, " => ", e.out_dims, ")")
end

outputsize(e::Embedding, _, ::AbstractRNG) = (e.out_dims,)

function (e::Embedding)(x::Number, ps, st::NamedTuple)
    @assert Utils.eltype(x) <: Integer
    return view(ps.weight, :, x), st
end
function (e::Embedding)(x::AbstractVector, ps, st::NamedTuple)
    @assert Utils.eltype(x) <: Integer
    return NNlib.gather(ps.weight, x), st
end
function (e::Embedding)(x::AbstractArray, ps, st::NamedTuple)
    @assert Utils.eltype(x) <: Integer
    y, stₙ = e(vec(x), ps, st)
    return reshape(y, :, size(x)...), stₙ
end
function (e::Embedding)(x::NTuple{N, T}, ps, st::NamedTuple) where {N, T}
    @assert Utils.eltype(T) <: Integer
    return view(ps.weight, :, x...), st
end
function (e::Embedding)(x::NTuple{N, <:AbstractVector{T}}, ps, st::NamedTuple) where {N, T}
    @assert Utils.eltype(T) <: Integer
    @argcheck allequal(size, x) DimensionMismatch("Input vectors must have the same shape")
    return NNlib.gather(ps.weight, x...), st
end
function (e::Embedding)(x::NTuple{N, <:AbstractArray{T}}, ps, st::NamedTuple) where {N, T}
    @assert Utils.eltype(T) <: Integer
    @argcheck allequal(size, x) DimensionMismatch("Input arrays must have the same shape")
    y, stₙ = e(vec.(x), ps, st)
    return reshape(y, :, size(first(x))...), stₙ
end
function (e::Embedding)(::Tuple{}, _, ::NamedTuple)
    throw(ArgumentError("Input tuple must contain at least one element"))
end

"""
    SinusoidalPositionalEncoding(....)

Sinusoidal Positional Encoding. For details see [1].

## Arguments

## Keyword Arguments

## Input

## Returns

## Parameters

## States

## References

[1] Vaswani, A. "Attention is all you need." Advances in Neural Information Processing
Systems (2017).
"""
@concrete struct SinusoidalPositionalEncoding{T} <: AbstractLuxLayer
    log_min_freq::T
    log_max_freq::T
    dims <: IntegerType
    scale <: Real
    full_turns::Bool
end

function SinusoidalPositionalEncoding(dims::IntegerType; min_freq=0.0001f0, max_freq=1.0f0,
        scale=nothing, full_turns::Bool=false)
    T = promote_type(typeof(min_freq), typeof(max_freq))
    scale = scale === nothing ? T(√(2 / dims)) : T(scale)
    return SinusoidalPositionalEncoding(
        T(log(min_freq)), T(log(max_freq)), dims, scale, full_turns)
end

function initialstates(::AbstractRNG, spe::SinusoidalPositionalEncoding{T}) where {T}
    one_zero = range(T(1), T(0); length=spe.dims ÷ 2)
    sigmas = exp.(one_zero .* (spe.log_max_freq - spe.log_min_freq) .+ spe.log_min_freq)
    spe.full_turns && (@. sigmas *= 2π)
    return (; sigmas)
end

function (spe::SinusoidalPositionalEncoding)(x::AbstractArray, ps, st::NamedTuple)
    y = reshape(match_eltype(spe, ps, st, x), 1, size(x)...) .* st.sigmas
    z = vcat(sin.(y), cos.(y)) .* spe.scale
    return z, st
end
