# Initial design is based off of https://github.com/FluxML/Flux.jl/blob/942c6e5051b7a8cb064432d1f0604319497d5f09/src/outputsize.jl
module NilSizePropagation

using ArrayInterface: ArrayInterface
using ForwardDiff: ForwardDiff
using Random: Random
using Static: Static, StaticBool

# We need these to avoid ambiguities
using StaticArraysCore: StaticArraysCore

using LuxLib: LuxLib
using NNlib: NNlib

struct Nil <: Real end

const nil = Nil()

Nil(::T) where {T<:Real} = nil
Nil(::Nil) = nil
(::Type{T})(::Nil) where {T<:Real} = nil
Base.convert(::Type{Nil}, ::Real) = nil
Base.convert(::Type{T}, ::Nil) where {T<:Real} = zero(T)
Base.convert(::Type{Nil}, ::Nil) = nil

Base.Bool(::Nil) = throw(ArgumentError("`Bool` is not defined for `Nil`."))

const NIL_DUAL_ERROR_MSG = "`Nil` is incompatible with `Dual` numbers."

ForwardDiff.Dual(::Nil) = throw(ArgumentError(NIL_DUAL_ERROR_MSG))
ForwardDiff.Dual{T}(::Nil) where {T} = throw(ArgumentError(NIL_DUAL_ERROR_MSG))
ForwardDiff.Dual{T,V}(::Nil) where {T,V} = throw(ArgumentError(NIL_DUAL_ERROR_MSG))
function ForwardDiff.Dual{T,V,Tag}(::Nil) where {T,V,Tag}
    throw(ArgumentError(NIL_DUAL_ERROR_MSG))
end
function Base.convert(::Type{ForwardDiff.Dual{T,V,Tag}}, ::Nil) where {T,V,Tag}
    throw(ArgumentError(NIL_DUAL_ERROR_MSG))
end

const NIL_STATIC_ERROR_MSG = "`Nil` is incompatible with `Static` numbers."

function Base.convert(
    ::Type{Nil}, ::Union{StaticBool{N},Static.StaticFloat64{N},Static.StaticInt{N}}
) where {N}
    throw(ArgumentError(NIL_STATIC_ERROR_MSG))
end

Base.float(::Type{Nil}) = Nil

for f in [
    :copy,
    :zero,
    :one,
    :oneunit,
    :+,
    :-,
    :abs,
    :abs2,
    :inv,
    :exp,
    :log,
    :log1p,
    :log2,
    :log10,
    :sqrt,
    :tanh,
    :conj,
]
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

Base.promote_rule(::Type{Nil}, ::Type{<:Real}) = Nil
function Base.promote_rule(::Type{Nil}, ::Type{ForwardDiff.Dual{T,V,Tag}}) where {T,V,Tag}
    throw(ArgumentError(NIL_DUAL_ERROR_MSG))
end

Base.rand(::Random.AbstractRNG, ::Random.SamplerType{Nil}) = nil

struct NilArray{N} <: AbstractArray{Nil,N}
    size::NTuple{N,Int}
end

NilArray(x::AbstractArray) = NilArray{ndims(x)}(size(x))

const AnyNilArray{N} = Union{NilArray{N},AbstractArray{<:Nil,N}}

function Base.show(io::IO, ::MIME"text/plain", x::AnyNilArray)
    Base.array_summary(io, x, axes(x))
    return nothing
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

function Base.similar(
    bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{NilArray}}, ::Type{ElType}
) where {ElType}
    return NilArray{ndims(bc)}(size(bc))
end

Base.copyto!(dest::NilArray, ::Broadcast.Broadcasted) = dest
function Base.copyto!(
    dest::NilArray, ::Broadcast.Broadcasted{<:StaticArraysCore.StaticArrayStyle}
)
    return dest
end
Base.copyto!(dest::NilArray, ::Broadcast.Broadcasted{Nothing}) = dest
function Base.copyto!(
    dest::NilArray, ::Broadcast.Broadcasted{<:Base.Broadcast.AbstractArrayStyle{0}}
)
    return dest
end
function Base.copyto!(
    dest::NilArray, ::Broadcast.Broadcasted{<:StaticArraysCore.StaticArrayStyle{0}}
)
    return dest
end

Base.fill!(dest::NilArray, _) = dest

const Optional{T} = Union{Nothing,T}
const Numeric = Union{<:Number,<:AbstractArray{<:Number}}

# Now we special case
# NOTE: If we don't define these, we are still good to go, but they take slightly longer
## Convolutions / Pooling
for N in (3, 4, 5)
    @eval function NNlib.conv!(
        y::AnyNilArray{$N},
        x::AbstractArray{<:Number,$N},
        w::AbstractArray{<:Number,$N},
        cdims::NNlib.DenseConvDims;
        kwargs...,
    )
        return y
    end
    for op in (:maxpool!, :meanpool!, :lpnormpool!)
        @eval function NNlib.$(op)(
            y::AnyNilArray{$N},
            x::AbstractArray{<:Number,$N},
            cdims::NNlib.PoolDims;
            kwargs...,
        )
            return y
        end
    end
end

## Normalization
function LuxLib.Impl.batchnorm(
    x::AnyNilArray{N},
    ::Optional{<:AbstractVector},
    ::Optional{<:AbstractVector},
    rμ::Optional{<:AbstractVector},
    rσ²::Optional{<:AbstractVector},
    ::StaticBool,
    act::F,
    ::Number,
    ::Number,
) where {N,F}
    return x, rμ, rσ²
end

function LuxLib.Impl.groupnorm(
    x::AnyNilArray{N},
    ::Optional{<:AbstractVector},
    ::Optional{<:AbstractVector},
    ::Int,
    act::F,
    ::Number,
) where {N,F}
    return x
end

function LuxLib.Impl.normalization(
    x::AnyNilArray,
    rμ::Optional{<:AbstractVector},
    rσ²::Optional{<:AbstractVector},
    ::Optional{<:AbstractVector},
    ::Optional{<:AbstractVector},
    _,
    ::StaticBool,
    __,
    ___,
    act::F,
) where {F}
    return x, rμ, rσ²
end

function LuxLib.Impl.affine_normalize(
    ::F, x::AnyNilArray, ::Numeric, ::Numeric, ::Nothing, ::Nothing, ::Number
) where {F}
    return x
end
function LuxLib.Impl.affine_normalize(
    ::F, x::AnyNilArray, ::Numeric, ::Numeric, ::AbstractArray, ::AbstractArray, ::Number
) where {F}
    return x
end

end

function LuxCore.outputsize(layer::AbstractLuxLayer, x, rng::AbstractRNG)
    ps, st = setup(rng, layer)
    fn = xᵢ -> xᵢ isa AbstractArray ? NilSizePropagation.NilArray(xᵢ) : xᵢ
    x_nil = Functors.fmap(fn, x)
    ps_nil = Functors.fmap(fn, ps)
    st_nil = Functors.fmap(fn, st)
    y = first(apply(layer, x_nil, ps_nil, st_nil))
    return Utils.unbatched_structure(y)
end
