"""
    Dropout(p; dims=:, initial_seed::UInt64=UInt64(0))

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

# # FIXME: Update to use RNGs
# """
#     VariationalHiddenDropout(p; dims=:, initial_seed::UInt64=UInt64(0))

# VariationalHiddenDropout layer. The only difference from Dropout is that the `mask` is retained until `EFL.update_state(l, :update_mask, true)` is called.

# # Arguments

# * To apply dropout along certain dimension(s), specify the `dims` keyword. e.g. `Dropout(p; dims = 3)` will randomly zero out entire channels on WHCN input (also called 2D dropout).
# * Each execution of the Layer increments the `seed` and returns it wrapped in the state

# Call [`testmode`](@ref) to switch to test mode.
# """
# struct VariationalHiddenDropout{T,D} <: AbstractExplicitLayer
#     p::T
#     initial_seed::UInt64
#     dims::D
# end

# initialstates(::AbstractRNG, d::VariationalHiddenDropout) = (seed=d.initial_seed, training=nothing, update_mask=true)

# function VariationalHiddenDropout(p; dims=:, initial_seed::UInt64=UInt64(0))
#     @assert 0 ≤ p ≤ 1
#     return VariationalHiddenDropout(p, initial_seed, dims)
# end

# function (d::VariationalHiddenDropout{T})(
#     x::AbstractArray{T}, ps, st::NamedTuple
# ) where {T}
#     !istraining(st) || return (x, st)
#     if st.update_mask
#         y, mask = dropout(MersenneTwister(st.seed), x, d.p; dims=d.dims)
#         return y, (mask=mask, seed=st.seed + 1, update_mask=false, training=st.training)
#     else
#         return x .* st.mask, st
#     end
# end

# function Base.show(io::IO, d::VariationalHiddenDropout)
#     print(io, "VariationalHiddenDropout(", d.p)
#     d.dims != Colon() && print(io, ", dims=", d.dims)
#     return print(io, ")")
# end
