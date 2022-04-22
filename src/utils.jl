replicate(rng::AbstractRNG) = copy(rng)
replicate(rng::CUDA.RNG) = deepcopy(rng)

zeros32(rng::AbstractRNG, args...; kwargs...) = zeros32(args...; kwargs...)
ones32(rng::AbstractRNG, args...; kwargs...) = ones32(args...; kwargs...)
Base.zeros(rng::AbstractRNG, args...; kwargs...) = zeros(args...; kwargs...)
Base.ones(rng::AbstractRNG, args...; kwargs...) = ones(args...; kwargs...)

@inline istraining() = false
@inline istraining(st::NamedTuple)::Bool = st.training

@inline _norm(x; dims=Colon()) = sqrt.(sum(abs2, x; dims=dims))

# Compute norm over all dimensions except `except_dim`
@inline _norm_except(x::AbstractArray{T,N}, except_dim) where {T,N} = _norm(x; dims=filter(i -> i != except_dim, 1:N))
@inline _norm_except(x::AbstractArray{T,N}) where {T,N} = _norm_except(x, N)

# Handling ComponentArrays
gpu(c::ComponentArray) = ComponentArray(gpu(getdata(c)), getaxes(c))
cpu(c::ComponentArray) = ComponentArray(cpu(getdata(c)), getaxes(c))

Base.zero(c::ComponentArray{T,N,<:CuArray{T}}) where {T,N} = ComponentArray(zero(getdata(c)), getaxes(c))

Base.vec(c::ComponentArray{T,N,<:CuArray{T}}) where {T,N} = getdata(c)

Base.:-(x::ComponentArray{T,N,<:CuArray{T}}) where {T,N} = ComponentArray(-getdata(x), getaxes(x))

function Base.similar(c::ComponentArray{T,N,<:CuArray{T}}, l::Vararg{Union{Integer,AbstractUnitRange}}) where {T,N}
    return similar(getdata(c), l)
end

function Zygote.accum(x::ComponentArray, ys::ComponentArray...)
    return ComponentArray(Zygote.accum(getdata(x), getdata.(ys)...), getaxes(x))
end

function Functors.functor(::Type{<:ComponentArray}, c)
    return NamedTuple{propertynames(c)}(getproperty.((c,), propertynames(c))), ComponentArray
end

function Optimisers.update!(st, ps::ComponentArray, gs::ComponentArray)
    Optimisers.update!(st, NamedTuple(ps), NamedTuple(gs))
    return st, ps
end

function ComponentArrays.make_carray_args(nt::NamedTuple)
    data, ax = ComponentArrays.make_carray_args(Vector, nt)
    data = length(data) == 0 ? Float32[] : (length(data)==1 ? [data[1]] : reduce(vcat, data))
    return (data, ax)
end

## For being able to print empty ComponentArrays
function ComponentArrays.last_index(f::FlatAxis)
    nt = ComponentArrays.indexmap(f)
    length(nt) == 0 && return 0
    return ComponentArrays.last_index(last(nt))
end

ComponentArrays.recursive_length(nt::NamedTuple{(), Tuple{}}) = 0

# Return Nothing if field not present
function safegetproperty(x::Union{ComponentArray,NamedTuple}, k::Symbol)
    k ∈ propertynames(x) && return getproperty(x, k)
    return nothing
end

# Getting typename
get_typename(::T) where {T} = Base.typename(T).wrapper
