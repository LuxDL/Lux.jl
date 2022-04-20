# Basic Dropout
struct Dropout{T,D} <: AbstractExplicitLayer
    p::T
    initial_seed::UInt64
    dims::D
end

initialstates(::AbstractRNG, d::Dropout) = (seed=d.initial_seed, training=:auto)

function Dropout(p; dims=:, initial_seed::UInt64=UInt64(0))
    @assert 0 ≤ p ≤ 1
    return Dropout(p, initial_seed, dims)
end

Base.@pure function (d::Dropout{T})(x::AbstractArray{T}, ps, st::NamedTuple) where {T}
    !istraining(st) || return (x, st)
    y = dropout(MersenneTwister(st.seed), x, d.p; dims=d.dims)[1]
    @set! st.seed = st.seed + 1
    return y, st
end

function Base.show(io::IO, d::Dropout)
    print(io, "Dropout(", d.p)
    if d.dims != Colon()
        print(io, ", dims=", d.dims)
    end
    return print(io, ")")
end

# Variational Hidden Dropout
## Mask is retained unless explicitly dropped
struct VariationalHiddenDropout{T,D} <: AbstractExplicitLayer
    p::T
    initial_seed::UInt64
    dims::D
end

initialstates(::AbstractRNG, d::VariationalHiddenDropout) = (seed=d.initial_seed, training=:auto, update_mask=true)

function VariationalHiddenDropout(p; dims=:, initial_seed::UInt64=UInt64(0))
    @assert 0 ≤ p ≤ 1
    return VariationalHiddenDropout(p, initial_seed, dims)
end

Base.@pure function (d::VariationalHiddenDropout{T})(
    x::AbstractArray{T}, ps, st::NamedTuple
) where {T}
    !istraining(st) || return (x, st)
    if st.update_mask
        y, mask = dropout(MersenneTwister(st.seed), x, d.p; dims=d.dims)
        return y, (mask=mask, seed=st.seed + 1, update_mask=false, training=st.training)
    else
        y = dropout(MersenneTwister(st.seed), st.mask, x, d.p; dims=d.dims)
        return y, st
    end
end
