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

@trace function (e::Embedding)(x::Union{Number,AbstractVector}, ps, st::NamedTuple)
    @assert Utils.eltype(x) <: Integer
    return ps.weight[:, x], st
end
@trace function (e::Embedding)(x::AbstractArray, ps, st::NamedTuple)
    @assert Utils.eltype(x) <: Integer
    y, stₙ = e(Utils.vec(x), ps, st)
    return reshape(y, :, size(x)...), stₙ
end
@trace function (e::Embedding)(x::NTuple{N,T}, ps, st::NamedTuple) where {N,T}
    @assert Utils.eltype(T) <: Integer
    return ps.weight[:, x...], st
end
@trace function (e::Embedding)(
    x::NTuple{N,<:AbstractVector{T}}, ps, st::NamedTuple
) where {N,T}
    @assert Utils.eltype(T) <: Integer
    allequal(size, x) || throw(DimensionMismatch("Input vectors must have the same shape"))
    return NNlib.gather(ps.weight, x...), st
end
@trace function (e::Embedding)(
    x::NTuple{N,<:AbstractArray{T}}, ps, st::NamedTuple
) where {N,T}
    @assert Utils.eltype(T) <: Integer
    allequal(size, x) || throw(DimensionMismatch("Input arrays must have the same shape"))
    y, stₙ = e(vec.(x), ps, st)
    return reshape(y, :, size(first(x))...), stₙ
end
@trace function (e::Embedding)(::Tuple{}, _, ::NamedTuple)
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

@trace function (spe::SinusoidalPositionalEmbedding)(x::AbstractArray, ps, st::NamedTuple)
    y = reshape(match_eltype(spe, ps, st, x), 1, size(x)...) .* st.sigmas
    z = vcat(sin.(y), cos.(y)) .* spe.scale
    return z, st
end

"""
    RotaryPositionalEmbedding(
        dim::IntegerType;
        max_sequence_length::IntegerType=4096,
        base::IntegerType=10000,
        low_memory_variant::Bool=true,
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
  - `low_memory_variant`: If `true` then cos and sin cache have leading dimension of
    `dim ÷ 2`. If `false` then cos and sin cache have leading dimension of `dim`.
    Default: `true`.

## Input

  - 4D `AbstractArray` such that

      + `size(x, 1) == dim`
      + `size(x, 3) ≤ max_sequence_length`

## Returns

  - 4D `AbstractArray` of the same size as the input.

## States

  - NamedTuple containing `cos_cache` and `sin_cache`.
"""
@concrete struct RotaryPositionalEmbedding <: AbstractLuxLayer
    dim <: IntegerType
    max_sequence_length <: IntegerType
    base <: IntegerType
    low_memory_variant::Bool
end

function RotaryPositionalEmbedding(
    dim::IntegerType;
    max_sequence_length::IntegerType=4096,
    base::IntegerType=10000,
    low_memory_variant::Bool=true,
)
    return RotaryPositionalEmbedding(dim, max_sequence_length, base, low_memory_variant)
end

function initialstates(::AbstractRNG, rope::RotaryPositionalEmbedding)
    return compute_rotary_embedding_params(
        rope.dim,
        rope.max_sequence_length;
        rope.base,
        dtype=Float32,
        rope.low_memory_variant,
    )
end

@trace function (rope::RotaryPositionalEmbedding)(
    x::AbstractArray{T,4}, ps, st::NamedTuple
) where {T}
    y = apply_rotary_embedding(x, st.cos_cache, st.sin_cache; seq_dim=3)
    return y, st
end

@trace function (rope::RotaryPositionalEmbedding)((x, input_pos)::Tuple, ps, st::NamedTuple)
    y = apply_rotary_embedding(x, input_pos, st.cos_cache, st.sin_cache; seq_dim=3)
    return y, st
end

## Functional variants since Qwen3 like models tend to share the same rotary embedding
## parameters
"""
    compute_rotary_embedding_params(head_dim::Integer, max_sequence_length::Integer;
                                    base::Number, dtype::Type{T}=Float32,
                                    low_memory_variant::Bool=true)

Computes the cosine and sine cache for rotary positional embeddings.

## Arguments

  - `head_dim`: The feature dimensions to be rotated.
  - `max_sequence_length`: The maximum sequence length. Default: `4096`.

## Keyword Arguments

  - `base`: The base used to compute angular frequency for each dimension in the positional
    encodings. Default: `10000`.
  - `dtype`: The data type of the cache. Default: `Float32`.
  - `low_memory_variant`: If `true` then cos and sin cache have leading dimension of
    `head_dim ÷ 2`. If `false` then cos and sin cache have leading dimension of `head_dim`.
    Default: `true`.
"""
function compute_rotary_embedding_params(
    head_dim::Integer,
    max_sequence_length::Integer;
    base::Number,
    dtype::Type{T}=Float32,
    low_memory_variant::Bool=true,
) where {T}
    θ = inv.(T.(base .^ (range(0, head_dim - 1; step=2)[1:(head_dim ÷ 2)] ./ head_dim)))
    seq_idx = collect(T, 0:(max_sequence_length - 1))
    angles = reshape(θ, :, 1) .* reshape(seq_idx, 1, :)
    low_memory_variant || (angles = vcat(angles, angles))
    return (; cos_cache=cos.(angles), sin_cache=sin.(angles))
end

"""
    apply_rotary_embedding(x::AbstractArray{T,4}, cos_cache::AbstractMatrix,
                           sin_cache::AbstractMatrix; head_dim::Integer, seq_dim::Integer)
    apply_rotary_embedding(x::AbstractArray{T,4}, input_positions::AbstractVector{<:Integer},
                           cos_cache::AbstractMatrix, sin_cache::AbstractMatrix;
                           head_dim::Integer, seq_dim::Integer)

Apply rotary embedding to the input `x` using the `cos_cache` and `sin_cache` parameters.
If `input_positions` is provided, then we extract the cosine and sine cache for the
corresponding positions in the sequence. Otherwise, we use the entire cache upto
sequence length of `x`.

## Arguments

  - `x`: 4D `AbstractArray`.
  - `cos_cache`: Cache of cosine values. Generated using
    [`compute_rotary_embedding_params`](@ref).
  - `sin_cache`: Cache of sine values. Generated using
    [`compute_rotary_embedding_params`](@ref).
  - `seq_dim`: Dimension of the sequence. Must be between 1 and 4.
  - `input_positions`: Positions in the sequence to extract the cosine and sine cache for.
    If not provided, then we use the entire cache upto sequence length of `x`.

## Returns

  - Output of the rotary embedding.
"""
function apply_rotary_embedding end

@trace function apply_rotary_embedding(
    x::AbstractArray{T,4},
    cos_cache::AbstractMatrix,
    sin_cache::AbstractMatrix;
    seq_dim::Integer,
    kwargs...,
) where {T}
    @assert 1 ≤ seq_dim ≤ 4 "seq_dim must be between 1 and 4"
    return apply_rotary_embedding(
        x, 1:size(x, seq_dim), cos_cache, sin_cache; seq_dim, kwargs...
    )
end

@trace function apply_rotary_embedding(
    x::AbstractArray{T,4},
    input_positions::AbstractVector{<:Integer},
    cos_cache::AbstractMatrix,
    sin_cache::AbstractMatrix;
    seq_dim::Integer,
) where {T}
    @assert seq_dim != 1 "seq_dim cannot be the first dimension"
    @assert 1 ≤ seq_dim ≤ 4 "seq_dim must be between 1 and 4"
    @assert size(cos_cache) == size(sin_cache)

    h_d, seq_len = size(x, 1), size(x, seq_dim)
    h_d_2 = h_d ÷ 2
    c_1, c_2 = size(cos_cache)

    if c_1 == h_d
        low_memory_variant = false
    elseif c_1 == h_d_2
        low_memory_variant = true
    else
        throw(DimensionMismatch("Input Dimension Mismatch: Expected $(h_d) or $(h_d_2), \
                                 got $(c_1)"))
    end

    @assert seq_len ≤ c_2 "Sequence length ($seq_len) must be less than $(c_2)"
    @assert seq_len == length(input_positions) "Sequence length ($seq_len) must be equal \
                                                  to length of input_positions ($seq_len)"

    cos_cache_reshaped, sin_cache_reshaped = standardize_cache_size(
        @view(cos_cache[:, input_positions]), @view(sin_cache[:, input_positions]), seq_dim
    )

    # split x into first half and second half
    x1 = @view x[1:h_d_2, :, :, :]
    x2 = @view x[(h_d_2 + 1):h_d, :, :, :]

    if low_memory_variant
        first_half = @. x1 * cos_cache_reshaped - x2 * sin_cache_reshaped
        second_half = @. x2 * cos_cache_reshaped + x1 * sin_cache_reshaped
        return vcat(first_half, second_half)
    else
        x_rotated = vcat(-x2, x1)
        return @. x * cos_cache_reshaped + x_rotated * sin_cache_reshaped
    end
end

function standardize_cache_size(
    cos_cache::AbstractMatrix, sin_cache::AbstractMatrix, seq_dim::Integer
)
    c_1, c_2 = size(cos_cache)
    final_shape = ntuple(i -> i == 1 ? c_1 : (i == seq_dim ? c_2 : 1), 4)
    return reshape(cos_cache, final_shape...), reshape(sin_cache, final_shape...)
end
