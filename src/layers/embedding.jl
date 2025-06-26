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

!!! warning "Gradients with Tracker.jl"

    Tracker.jl produces incorrect gradients for this layer if indices in the input are
    repeated. Don't use this layer with Tracker.jl if you need to compute gradients.
"""
@concrete struct Embedding <: AbstractLuxLayer
    in_dims <: Union{IntegerType,Tuple{Vararg{IntegerType}}}
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

function (e::Embedding)(x::Union{Number,AbstractVector}, ps, st::NamedTuple)
    @argcheck Utils.eltype(x) <: Integer
    return ps.weight[:, x], st
end
function (e::Embedding)(x::AbstractArray, ps, st::NamedTuple)
    @argcheck Utils.eltype(x) <: Integer
    y, stₙ = e(Utils.vec(x), ps, st)
    return reshape(y, :, size(x)...), stₙ
end
function (e::Embedding)(x::NTuple{N,T}, ps, st::NamedTuple) where {N,T}
    @argcheck Utils.eltype(T) <: Integer
    return ps.weight[:, x...], st
end
function (e::Embedding)(x::NTuple{N,<:AbstractVector{T}}, ps, st::NamedTuple) where {N,T}
    @argcheck Utils.eltype(T) <: Integer
    @argcheck allequal(size, x) DimensionMismatch("Input vectors must have the same shape")
    return NNlib.gather(ps.weight, x...), st
end
function (e::Embedding)(x::NTuple{N,<:AbstractArray{T}}, ps, st::NamedTuple) where {N,T}
    @argcheck Utils.eltype(T) <: Integer
    @argcheck allequal(size, x) DimensionMismatch("Input arrays must have the same shape")
    y, stₙ = e(vec.(x), ps, st)
    return reshape(y, :, size(first(x))...), stₙ
end
function (e::Embedding)(::Tuple{}, _, ::NamedTuple)
    throw(ArgumentError("Input tuple must contain at least one element"))
end

@doc doc"""
    SinusoidalPositionalEmbedding(
        dims::IntegerType; min_freq=0.0001f0, max_freq=1.0f0,
        scale=nothing, full_turns::Bool=false
    )

Sinusoidal Positional Embedding. For details see [vaswani2017attention](@citet).

## Arguments

  - `dims`: The dimensionality of the resulting positional embeddings.

## Keyword Arguments

  - `min_freq`: The minimum frequency expected. Default: `0.0001f0`.
  - `max_freq`: The maximum frequency expected. Default: `1.0f0`.
  - `scale`: A multiplicative scale for the embeddings. Default: $\sqrt{2/dims}$.
  - `full_turns`: If `true` multiply the frequencies with $2\pi$. Default: `false`.

## Input

  - AbstractArray

## Returns

  - If the input array is of size `(insz...,)` then the output is of size `(dims, insz...)`.

## States

  - NamedTuple containing `sigmas`.
"""
@concrete struct SinusoidalPositionalEmbedding{T} <: AbstractLuxLayer
    log_min_freq::T
    log_max_freq::T
    dims <: IntegerType
    scale <: Real
    full_turns::Bool
end

function SinusoidalPositionalEmbedding(
    dims::IntegerType;
    min_freq=0.0001f0,
    max_freq=1.0f0,
    scale=nothing,
    full_turns::Bool=false,
)
    T = promote_type(typeof(min_freq), typeof(max_freq))
    scale = scale === nothing ? T(√(2 / dims)) : T(scale)
    return SinusoidalPositionalEmbedding(
        T(log(min_freq)), T(log(max_freq)), dims, scale, full_turns
    )
end

function initialstates(::AbstractRNG, spe::SinusoidalPositionalEmbedding{T}) where {T}
    one_zero = range(T(1), T(0); length=spe.dims ÷ 2)
    sigmas = exp.(one_zero .* (spe.log_max_freq - spe.log_min_freq) .+ spe.log_min_freq)
    spe.full_turns && (@. sigmas *= 2π)
    return (; sigmas)
end

function (spe::SinusoidalPositionalEmbedding)(x::AbstractArray, ps, st::NamedTuple)
    y = reshape(match_eltype(spe, ps, st, x), 1, size(x)...) .* st.sigmas
    z = vcat(sin.(y), cos.(y)) .* spe.scale
    return z, st
end

"""
    RotaryPositionalEmbedding(
        dim::IntegerType; max_sequence_length::IntegerType=4096, base::IntegerType=10000
    )

Rotary Positional Embedding. For details see [su2024roformer](@citet).

The traditional implementation rotates consecutive pairs of elements in the feature
dimension while the default implementation rotates pairs with stride half the feature
dimensions for efficiency.

## Arguments

  - `dim`: The feature dimensions to be rotated. If the input feature is larger than dims
    then the rest is left unchanged.

## Keyword Arguments

  - `base`: The base used to compute angular frequency for each dimension in the positional
    encodings. Default: `10000`.
  - `max_sequence_length`: The maximum sequence length. Default: `4096`.

## Input

  - 4D AbstractArray such that

      + `size(x, 1) == dim`
      + `size(x, 3) ≤ max_sequence_length`

## Returns

  - 4D AbstractArray of the same size as the input.

## States

  - NamedTuple containing `cos_cache` and `sin_cache`.
"""
@concrete struct RotaryPositionalEmbedding <: AbstractLuxLayer
    dim <: IntegerType
    max_sequence_length <: IntegerType
    base <: IntegerType
end

function RotaryPositionalEmbedding(
    dim::IntegerType; max_sequence_length::IntegerType=4096, base::IntegerType=10000
)
    return RotaryPositionalEmbedding(dim, max_sequence_length, base)
end

function initialstates(::AbstractRNG, rope::RotaryPositionalEmbedding)
    theta =
        inv.(
            Float32.(
                rope.base .^ (range(0, rope.dim - 1; step=2)[1:(rope.dim ÷ 2)] ./ rope.dim)
            )
        )

    seq_idx = collect(Float32, 0:(rope.max_sequence_length - 1))
    idx_theta = reshape(theta, :, 1) .* reshape(seq_idx, 1, :)
    return (; cos_cache=cos.(idx_theta), sin_cache=sin.(idx_theta))
end

function (rope::RotaryPositionalEmbedding)(
    x::AbstractArray{T,4}, ps, st::NamedTuple
) where {T}
    return rope((x, nothing), ps, st)
end

function (rope::RotaryPositionalEmbedding)((x, input_pos)::Tuple, ps, st::NamedTuple)
    @argcheck ndims(x) == 4 "Input must be a 4D tensor ([dim, nheads, seq_len, batch])"
    @argcheck size(x, 3) ≤ rope.max_sequence_length "Sequence length must be less than \
                                                     $(rope.max_sequence_length)"
    @argcheck size(x, 1) == rope.dim "Input Dimension Mismatch: Expected $(rope.dim), got \
                                      $(size(x, 1))"

    h_d, _, seq_len, _ = size(x)
    y = match_eltype(rope, ps, st, x)

    # extract the values based on whether input_pos is set or not
    if input_pos === nothing
        cos_cache = st.cos_cache[:, 1:seq_len]
        sin_cache = st.sin_cache[:, 1:seq_len]
    else
        cos_cache = st.cos_cache[:, input_pos]
        sin_cache = st.sin_cache[:, input_pos]
    end

    x_shaped1 = y[1:(h_d ÷ 2), :, :, :]
    x_shaped2 = y[(h_d ÷ 2 + 1):end, :, :, :]

    # reshape the cache for broadcasting
    cos_cache = reshape(cos_cache, :, 1, seq_len, 1)
    sin_cache = reshape(sin_cache, :, 1, seq_len, 1)

    x_out = vcat(
        x_shaped1 .* cos_cache - x_shaped2 .* sin_cache,
        x_shaped2 .* cos_cache + x_shaped1 .* sin_cache,
    )

    return x_out, st
end
