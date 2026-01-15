"""
    AlphaDropout(p::Real)

AlphaDropout layer.

## Arguments

  - `p`: Probability of Dropout

      + if `p = 0` then [`NoOpLayer`](@ref) is returned.
      + if `p = 1` then `WrappedLayer(Base.Fix1(broadcast, zero))` is returned.

## Inputs

  - `x`: Must be an AbstractArray

## Returns

  - `x` with dropout mask applied if `training=Val(true)` else just `x`
  - State with updated `rng`

## States

  - `rng`: Pseudo Random Number Generator
  - `training`: Used to check if training/inference mode

Call [`Lux.testmode`](@ref) to switch to test mode.

See also [`Lux.Dropout`](@ref), [`VariationalHiddenDropout`](@ref)
"""
struct AlphaDropout{T<:Real} <: AbstractLuxLayer
    p::T
    alpha::T
    scale::T
    bias::T
end

function initialstates(rng::AbstractRNG, ::AlphaDropout)
    return (rng=Utils.sample_replicate(rng), training=Val(true))
end

function AlphaDropout(p::T) where {T<:Real}
    @assert 0 ≤ p ≤ 1
    iszero(p) && return NoOpLayer()
    isone(p) && return WrappedFunction(Base.Fix1(broadcast, zero))

    α = T(-1.7580993408473766)
    γ = T(inv(sqrt((1 - p) * (1 + p * α^2))))
    β = T(-γ * α * p)
    return AlphaDropout(p, α, γ, β)
end

function (d::AlphaDropout)(x, _, st::NamedTuple)
    y, rng = alpha_dropout(st.rng, x, d.p, st.training, d.alpha, d.scale, d.bias)
    return y, (; rng, st.training)
end

Base.show(io::IO, d::AlphaDropout) = print(io, "AlphaDropout(", d.p, ")")

"""
    Dropout(p; dims=:)

Dropout layer.

## Arguments

  - `p`: Probability of Dropout (if `p = 0` then [`NoOpLayer`](@ref) is returned)

## Keyword Arguments

  - To apply dropout along certain dimension(s), specify the `dims` keyword. e.g.
    `Dropout(p; dims = (3,4))` will randomly zero out entire channels on WHCN input
    (also called 2D dropout).

## Inputs

  - `x`: Must be an AbstractArray

## Returns

  - `x` with dropout mask applied if `training=Val(true)` else just `x`
  - State with updated `rng`

## States

  - `rng`: Pseudo Random Number Generator
  - `training`: Used to check if training/inference mode

Call [`Lux.testmode`](@ref) to switch to test mode.

See also [`AlphaDropout`](@ref), [`VariationalHiddenDropout`](@ref)
"""
@concrete struct Dropout{T} <: AbstractLuxLayer
    p::T
    q::T
    dims
end

function initialstates(rng::AbstractRNG, ::Dropout)
    return (rng=Utils.sample_replicate(rng), training=Val(true))
end

function Dropout(p; dims=:)
    @assert 0 ≤ p ≤ 1
    iszero(p) && return NoOpLayer()
    return Dropout(p, 1 / (1 - p), dims)
end

function (d::Dropout)(x, _, st::NamedTuple)
    y, _, rng = dropout(st.rng, x, d.p, st.training, d.q, d.dims)
    return y, (; rng, st.training)
end

function Base.show(io::IO, d::Dropout)
    print(io, "Dropout(", d.p)
    d.dims != Colon() && print(io, ", dims=", d.dims)
    return print(io, ")")
end

"""
    VariationalHiddenDropout(p; dims=:)

VariationalHiddenDropout layer. The only difference from Dropout is that the `mask` is
retained until [`Lux.update_state(l, :update_mask, Val(true))`](@ref) is called.

## Arguments

  - `p`: Probability of Dropout (if `p = 0` then [`NoOpLayer`](@ref) is returned)

## Keyword Arguments

  - To apply dropout along certain dimension(s), specify the `dims` keyword. e.g.
    `VariationalHiddenDropout(p; dims = 3)` will randomly zero out entire channels on WHCN
    input (also called 2D dropout).

## Inputs

  - `x`: Must be an AbstractArray

## Returns

  - `x` with dropout mask applied if `training=Val(true)` else just `x`
  - State with updated `rng`

## States

  - `rng`: Pseudo Random Number Generator
  - `training`: Used to check if training/inference mode
  - `mask`: Dropout mask. Initilly set to nothing. After every run, contains the mask
    applied in that call
  - `update_mask`: Stores whether new mask needs to be generated in the current call

Call [`Lux.testmode`](@ref) to switch to test mode.

See also [`AlphaDropout`](@ref), [`Lux.Dropout`](@ref)
"""
@concrete struct VariationalHiddenDropout{T} <: AbstractLuxLayer
    p::T
    q::T
    dims
end

function initialstates(rng::AbstractRNG, ::VariationalHiddenDropout)
    return (
        rng=Utils.sample_replicate(rng),
        training=Val(true),
        update_mask=Val(true),
        mask=nothing,
    )
end

preserves_state_type(::VariationalHiddenDropout) = false

function VariationalHiddenDropout(p; dims=:)
    @assert 0 ≤ p ≤ 1
    iszero(p) && return NoOpLayer()
    return VariationalHiddenDropout(p, 1 / (1 - p), dims)
end

function (d::VariationalHiddenDropout)(x, _, st::NamedTuple)
    maskₒ = st.mask === nothing ? x : st.mask
    y, mask, rng = dropout(st.rng, x, maskₒ, d.p, st.training, st.update_mask, d.q, d.dims)
    return y, merge(st, (; mask, rng, update_mask=Val(false)))
end

function Base.show(io::IO, ::MIME"text/plain", d::VariationalHiddenDropout)
    print(io, "VariationalHiddenDropout(", d.p)
    d.dims != Colon() && print(io, ", dims=", d.dims)
    return print(io, ")")
end
