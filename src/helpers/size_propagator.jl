# Initial design is based off of https://github.com/FluxML/Flux.jl/blob/942c6e5051b7a8cb064432d1f0604319497d5f09/src/outputsize.jl
# Currently this is not being used anywhere. However, with 1.0 release we will define
# outputsize for all layers using this.
module NilSizePropagation

using ArrayInterface: ArrayInterface
using ..Lux: recursive_map
using LuxLib: LuxLib
using NNlib: NNlib
using Random: Random
using Static: StaticBool

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

NilArray(x::AbstractArray) = NilArray{ndims(x)}(size(x))

const AnyNilArray{N} = Union{NilArray{N}, AbstractArray{<:Nil, N}}

function Base.show(io::IO, ::MIME"text/plain", x::AnyNilArray)
    Base.array_summary(io, x, axes(x))
    return
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

Base.fill!(dest::NilArray, _) = dest

Base.mapreducedim!(_, __, R::AnyNilArray, ::Base.AbstractArrayOrBroadcasted) = R

recursively_nillify_internal(x) = x
recursively_nillify_internal(x::AbstractArray) = NilArray(x)

recursively_nillify(x::AbstractArray{<:Number}) = recursively_nillify_internal(x)
recursively_nillify(x) = recursive_map(recursively_nillify_internal, x)

const Optional{T} = Union{Nothing, T}
const Numeric = Union{<:Number, <:AbstractArray{<:Number}}

# Now we special case
# NOTE: If we don't define these, we are still good to go, but they take slightly longer
## Convolutions / Pooling
for N in (3, 4, 5)
    @eval function NNlib.conv!(y::AnyNilArray{$N}, x::AbstractArray{<:Number, $N},
            w::AbstractArray{<:Number, $N}, cdims::NNlib.DenseConvDims; kwargs...)
        return y
    end
    for op in (:maxpool!, :meanpool!, :lpnormpool!)
        @eval function NNlib.$(op)(y::AnyNilArray{$N}, x::AbstractArray{<:Number, $N},
                cdims::NNlib.PoolDims; kwargs...)
            return y
        end
    end
end

## Normalization
function LuxLib.Impl.batchnorm(
        x::AnyNilArray{N}, ::Optional{<:AbstractVector}, ::Optional{<:AbstractVector},
        rμ::Optional{<:AbstractVector}, rσ²::Optional{<:AbstractVector},
        ::StaticBool, act::F, ::Real, ::Real) where {N, F}
    return x, rμ, rσ²
end

function LuxLib.Impl.groupnorm(x::AnyNilArray{N}, ::Optional{<:AbstractVector},
        ::Optional{<:AbstractVector}, ::Int, act::F, ::Real) where {N, F}
    return x
end

function LuxLib.Impl.normalization(x::AnyNilArray, rμ::Optional{<:AbstractVector},
        rσ²::Optional{<:AbstractVector}, ::Optional{<:AbstractVector},
        ::Optional{<:AbstractVector}, _, ::StaticBool, __, ___, act::F) where {F}
    return x, rμ, rσ²
end

function LuxLib.Impl.affine_normalize(
        ::F, x::AnyNilArray, ::Numeric, ::Numeric, ::Nothing, ::Nothing, ::Real) where {F}
    return x
end
function LuxLib.Impl.affine_normalize(::F, x::AnyNilArray, ::Numeric, ::Numeric,
        ::AbstractArray, ::AbstractArray, ::Real) where {F}
    return x
end

end

# TODO: In v1 we change to this `outputsize` function, till then this is private API
function compute_output_size(layer::AbstractExplicitLayer,
        input_size::NTuple{N, <:Integer}, rng::AbstractRNG) where {N}
    x = NilSizePropagation.NilArray{N}(input_size)
    return compute_output_size(layer, x, rng)
end

function compute_output_size(
        layer::AbstractExplicitLayer, input_size::NTuple{N, <:Integer}, ps, st) where {N}
    x = NilSizePropagation.NilArray{N}(input_size)
    return compute_output_size(layer, x, ps, st)
end

function compute_output_size(layer::AbstractExplicitLayer, x, rng::AbstractRNG)
    ps, st = setup(rng, layer)
    return compute_output_size(layer, x, ps, st)
end

function compute_output_size(layer::AbstractExplicitLayer, x, ps, st)
    x_nil = NilSizePropagation.recursively_nillify(x)
    ps_nil = NilSizePropagation.recursively_nillify(ps)
    st_nil = NilSizePropagation.recursively_nillify(st)
    y = first(apply(layer, x_nil, ps_nil, st_nil))
    return Utils.unbatched_structure(y)
end
