"""
    Dropout(p; dims=:)

Dropout layer.

# Arguments

* To apply dropout along certain dimension(s), specify the `dims` keyword. e.g. `Dropout(p; dims = 3)` will randomly zero out entire channels on WHCN input (also called 2D dropout).
* Each execution of the Layer increments the `seed` and returns it wrapped in the state

Call [`testmode`](@ref) to switch to test mode.
"""
struct Dropout{T,D} <: AbstractExplicitLayer
    p::T
    dims::D
end

initialstates(rng::AbstractRNG, ::Dropout) = (rng=rng, training=true)

function Dropout(p; dims=:)
    @assert 0 ≤ p ≤ 1
    return Dropout(p, dims)
end

function (d::Dropout{T})(x::AbstractArray{T}, ps, st::NamedTuple) where {T}
    y, _, rng = dropout(st.rng, x, d.p, d.dims, istraining(st))
    return y, merge(st, (rng=rng,))
end

function Base.show(io::IO, d::Dropout)
    print(io, "Dropout(", d.p)
    d.dims != Colon() && print(io, ", dims=", d.dims)
    return print(io, ")")
end

"""
    VariationalHiddenDropout(p; dims=:)

VariationalHiddenDropout layer. The only difference from Dropout is that the `mask` is retained until `EFL.update_state(l, :update_mask, true)` is called.

# Arguments

* To apply dropout along certain dimension(s), specify the `dims` keyword. e.g. `Dropout(p; dims = 3)` will randomly zero out entire channels on WHCN input (also called 2D dropout).
* Each execution of the Layer increments the `seed` and returns it wrapped in the state

Call [`testmode`](@ref) to switch to test mode.
"""
struct VariationalHiddenDropout{T,D} <: AbstractExplicitLayer
    p::T
    dims::D
end

initialstates(rng::AbstractRNG, d::VariationalHiddenDropout) = (rng=rng, training=true, update_mask=true)

function VariationalHiddenDropout(p; dims=:)
    @assert 0 ≤ p ≤ 1
    return VariationalHiddenDropout(p, dims)
end

function (d::VariationalHiddenDropout{T})(x::AbstractArray{T}, ps, st::NamedTuple) where {T}
    if st.update_mask
        y, mask, rng = dropout(st.rng, x, d.p, d.dims, istraining(st))
        return y, (mask=mask, rng=rng, update_mask=false, training=st.training)
    else
        if !istraining(st)
            return x, st
        end
        return applydropout(x, st.mask), st
    end
end

function Base.show(io::IO, d::VariationalHiddenDropout)
    print(io, "VariationalHiddenDropout(", d.p)
    d.dims != Colon() && print(io, ", dims=", d.dims)
    return print(io, ")")
end
