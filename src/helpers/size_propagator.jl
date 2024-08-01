# Initial design is based off of https://github.com/FluxML/Flux.jl/blob/942c6e5051b7a8cb064432d1f0604319497d5f09/src/outputsize.jl
struct Nil <: Real end

const nil = Nil()

Nil(::T) where {T <: Number} = nil
(::Type{T})(::Nil) where {T <: Number} = nil
Base.convert(::Type{Nil}, ::Number) = nil
Base.convert(::Type{T}, ::Nil) where {T <: Number} = zero(T)
Base.convert(::Type{Nil}, ::Nil) = nil

Base.float(::Type{Nil}) = Nil

for f in [:copy, :zero, :one, :oneunit, :+, :-, :abs, :abs2, :inv,
    :exp, :log, :log1p, :log2, :log10, :sqrt, :tanh, :conj]
    @eval Base.$f(::Nil) = nil
end

for f in [:+, :-, :*, :/, :^, :mod, :div, :rem]
    @eval Base.$f(::Nil, ::Nil) = nil
end

Base.:<(::Nil, ::Nil) = true
Base.:≤(::Nil, ::Nil) = true

Base.isnan(::Nil) = false
Base.isfinite(::Nil) = true
Base.typemin(::Type{Nil}) = nil
Base.typemax(::Type{Nil}) = nil

Base.promote_rule(::Type{Nil}, ::Type{<:Number}) = Nil

Random.rand(::Random.AbstractRNG, ::Random.SamplerType{Nil}) = nil

struct NilArray{N} <: AbstractArray{Nil, N}
    size::NTuple{N, Int}
end

ArrayInterface.can_setindex(::Type{NilArray}) = false  # technically we can setindex but preferable to not
ArrayInterface.fast_scalar_indexing(::Type{<:AbstractArray{<:Nil}}) = false
ArrayInterface.fast_scalar_indexing(::Type{NilArray}) = false

Base.size(x::NilArray) = x.size

Base.convert(::Type{NilArray}, x::AbstractArray) = NilArray(size(x))

Base.getindex(::NilArray, ::Int...) = nil

Base.setindex!(::NilArray, v, ::Int...) = nil

function Base.similar(::NilArray, ::Type{ElType}, dims::Dims) where {ElType}
    return NilArray{length(dims)}(dims)
end

Broadcast.BroadcastStyle(::Type{<:NilArray}) = Broadcast.ArrayStyle{NilArray}()
Broadcast.BroadcastStyle(::Type{<:AbstractArray{<:Nil}}) = Broadcast.ArrayStyle{NilArray}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{NilArray}},
        ::Type{ElType}) where {ElType}
    return NilArray{ndims(bc)}(size(bc))
end

Base.copyto!(dest::NilArray, ::Broadcast.Broadcasted) = dest

# Now we special case
## Convolutions
for N in (3, 4, 5)
    @eval function NNlib.conv!(y::NilArray{$N}, x::AbstractArray{<:Number, $N},
            w::AbstractArray{<:Number, $N}, cdims::NNlib.ConvDims; kwargs...)
        return y
    end
end
