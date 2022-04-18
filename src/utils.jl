zeros32(rng::AbstractRNG, args...; kwargs...) = zeros32(args...; kwargs...)
ones32(rng::AbstractRNG, args...; kwargs...) = ones32(args...; kwargs...)
Base.zeros(rng::AbstractRNG, args...; kwargs...) = zeros(args...; kwargs...)
Base.ones(rng::AbstractRNG, args...; kwargs...) = ones(args...; kwargs...)

function var!(y1, y2, x; kwargs...)
    mean!(y1, x)
    return var!(y1, y2, x, y1; kwargs...)
end

function var!(y1, y2, x, μ; corrected::Bool = true)
    m = (length(x) ÷ length(y1)) - corrected
    @. y2 = abs2(x - μ) / m
    mean!(y1, x)
    return y1
end

istraining() = false
istraining(st::NamedTuple)::Bool = st.training == :auto ? istraining() : st.training

@inline _norm(x; dims=Colon()) = sqrt.(sum(abs2, x; dims=dims))

# Compute norm over all dimensions except `except_dim`
@inline _norm_except(x::AbstractArray{T,N}, except_dim) where {T,N} = _norm(x; dims=filter(i -> i != except_dim, 1:N))
@inline _norm_except(x::AbstractArray{T,N}) where {T,N} = _norm_except(x, N)

# Handling ComponentArrays
gpu(c::ComponentArray) = ComponentArray(gpu(getdata(c)), getaxes(c))
cpu(c::ComponentArray) = ComponentArray(cpu(getdata(c)), getaxes(c))

Base.zero(c::ComponentArray{T,N,<:CuArray{T}}) where {T,N} = ComponentArray(zero(getdata(c)), getaxes(c))

Base.vec(c::ComponentArray{T,N,<:CuArray{T}}) where {T,N} = getdata(c)

function Base.similar(c::ComponentArray{T,N,<:CuArray{T}}, l::Vararg{Union{Integer, AbstractUnitRange}}) where {T,N}
    return similar(getdata(c), l)
end     

function Zygote.accum(x::ComponentArray, y::ComponentArray)
    return ComponentArray(Zygote.accum(getdata(x), getdata(y)), getaxes(x))
end

# Zygote things
unfill_array(x::Fill) = Array(x)
unfill_array(x::AbstractArray) = x