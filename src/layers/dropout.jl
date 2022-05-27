"""
    Dropout(p; dims=:)

Dropout layer.

## Arguments

* `p`: Probability of Dropout (if `p = 0` then [`NoOpLayer`](@ref) is returned)

## Keyword Arguments

* To apply dropout along certain dimension(s), specify the `dims` keyword. e.g. `Dropout(p; dims = 3)` will randomly zero out entire channels on WHCN input (also called 2D dropout).

## Inputs

* `x`: Must be an AbstractArray

## Returns

* `x` with dropout mask applied if `training=Val(true)` else just `x`
* State with updated `rng`

## States

* `rng`: Pseudo Random Number Generator
* `training`: Used to check if training/inference mode

Call [`Lux.testmode`](@ref) to switch to test mode.

See also [`VariationalHiddenDropout`](@ref)
"""
struct Dropout{T, D} <: AbstractExplicitLayer
    p::T
    q::T
    dims::D
end

function initialstates(rng::AbstractRNG, ::Dropout)
    randn(rng)
    return (rng=replicate(rng), training=Val(true))
end

function Dropout(p; dims=:)
    @assert 0 ≤ p ≤ 1
    iszero(p) && return NoOpLayer()
    return Dropout(p, 1 / (1 - p), dims)
end

function (d::Dropout{T})(x::AbstractArray{T}, ps, st::NamedTuple) where {T}
    y, _, rng = dropout(st.rng, x, d.p, d.q, d.dims, st.training)
    return y, merge(st, (rng=rng,))
end

function Base.show(io::IO, d::Dropout)
    print(io, "Dropout(", d.p)
    d.dims != Colon() && print(io, ", dims=", d.dims)
    return print(io, ")")
end

"""
    VariationalHiddenDropout(p; dims=:)

VariationalHiddenDropout layer. The only difference from Dropout is that the `mask` is retained until [`Lux.update_state(l, :update_mask, Val(true))`](@ref) is called.

## Arguments

* `p`: Probability of Dropout (if `p = 0` then [`NoOpLayer`](@ref) is returned)

## Keyword Arguments

* To apply dropout along certain dimension(s), specify the `dims` keyword. e.g. `VariationalHiddenDropout(p; dims = 3)` will randomly zero out entire channels on WHCN input (also called 2D dropout).

## Inputs

* `x`: Must be an AbstractArray

## Returns

* `x` with dropout mask applied if `training=Val(true)` else just `x`
* State with updated `rng`

## States

* `rng`: Pseudo Random Number Generator
* `training`: Used to check if training/inference mode
* `mask`: Dropout mask. Initilly set to nothing. After every run, contains the mask applied in that call
* `update_mask`: Stores whether new mask needs to be generated in the current call

Call [`Lux.testmode`](@ref) to switch to test mode.

See also [`Dropout`](@ref)
"""
struct VariationalHiddenDropout{T, D} <: AbstractExplicitLayer
    p::T
    q::T
    dims::D
end

function initialstates(rng::AbstractRNG, ::VariationalHiddenDropout)
    randn(rng)
    return (rng=replicate(rng), training=Val(true), update_mask=Val(true),
            mask=nothing)
end

function VariationalHiddenDropout(p; dims=:)
    @assert 0 ≤ p ≤ 1
    iszero(p) && return NoOpLayer()
    return VariationalHiddenDropout(p, 1 / (1 - p), dims)
end

function (d::VariationalHiddenDropout{T})(x::AbstractArray{T}, ps, st::NamedTuple) where {T}
    y, mask, rng, update_mask = dropout(st.rng, x, st.mask, d.p, d.q, d.dims, st.training,
                                        st.update_mask)
    return y, merge(st, (mask=mask, rng=rng, update_mask=update_mask))
end

function Base.show(io::IO, d::VariationalHiddenDropout)
    print(io, "VariationalHiddenDropout(", d.p)
    d.dims != Colon() && print(io, ", dims=", d.dims)
    return print(io, ")")
end
