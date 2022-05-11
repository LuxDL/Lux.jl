"""
    Dropout(p; dims=:)

Dropout layer.

# Arguments

* To apply dropout along certain dimension(s), specify the `dims` keyword. e.g. `Dropout(p; dims = 3)` will randomly zero out entire channels on WHCN input (also called 2D dropout).
* Each execution of the Layer increments the `seed` and returns it wrapped in the state

Call [`Lux.testmode`](@ref) to switch to test mode.
"""
struct Dropout{T,D} <: AbstractExplicitLayer
    p::T
    dims::D
end

function initialstates(rng::AbstractRNG, ::Dropout)
    # FIXME: Take PRNGs seriously
    randn(rng, 1)
    return (rng=replicate(rng), training=Val(true))
end

function Dropout(p; dims=:)
    @assert 0 ≤ p ≤ 1
    return Dropout(p, dims)
end

function (d::Dropout{T})(x::AbstractArray{T}, ps, st::NamedTuple) where {T}
    y, _, rng = dropout(st.rng, x, d.p, d.dims, st.training)
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

# Arguments

* To apply dropout along certain dimension(s), specify the `dims` keyword. e.g. `Dropout(p; dims = 3)` will randomly zero out entire channels on WHCN input (also called 2D dropout).
* Each execution of the Layer increments the `seed` and returns it wrapped in the state

Call [`Lux.testmode`](@ref) to switch to test mode.
"""
struct VariationalHiddenDropout{T,D} <: AbstractExplicitLayer
    p::T
    dims::D
end

function initialstates(rng::AbstractRNG, ::VariationalHiddenDropout)
    # FIXME: Take PRNGs seriously
    randn(rng, 1)
    return (rng=replicate(rng), training=Val(true), update_mask=Val(true), mask=nothing)
end

function VariationalHiddenDropout(p; dims=:)
    @assert 0 ≤ p ≤ 1
    return VariationalHiddenDropout(p, dims)
end

function (d::VariationalHiddenDropout{T})(x::AbstractArray{T}, ps, st::NamedTuple) where {T}
    y, mask, rng, update_mask = dropout(st.rng, x, st.mask, d.p, d.dims, st.training, st.update_mask)
    return y, merge(st, (mask=mask, rng=rng, update_mask=update_mask))
end

function Base.show(io::IO, d::VariationalHiddenDropout)
    print(io, "VariationalHiddenDropout(", d.p)
    d.dims != Colon() && print(io, ", dims=", d.dims)
    return print(io, ")")
end
