module Utils

using ArrayInterface: ArrayInterface
using ArgCheck: @argcheck
using ChainRulesCore: ChainRulesCore, @non_differentiable, NoTangent
using ConcreteStructs: @concrete
using EnzymeCore: EnzymeRules
using ForwardDiff: Dual
using Functors: fmapstructure
using Random: AbstractRNG
using Static: Static, StaticBool, StaticInteger, StaticSymbol
using StaticArraysCore: SMatrix, SVector

using LuxCore: LuxCore, AbstractLuxLayer
using NNlib: NNlib

const CRC = ChainRulesCore

const BoolType = Union{StaticBool, Bool, Val{true}, Val{false}}
const IntegerType = Union{Integer, StaticInteger}
const SymbolType = Union{Symbol, StaticSymbol, Val}

# Aliased `size` from Base
size(x::AbstractArray) = Base.size(x)
size(x::T) where {T} = hasmethod(Base.size, Tuple{T}) ? Base.size(x) : nothing

@non_differentiable size(::Any)

structure(x) = fmapstructure(size, x)

size_unbatched(x::AbstractVector) = Base.size(x)
size_unbatched(x::AbstractArray) = Base.size(x)[1:(end - 1)]
function size_unbatched(x::T) where {T}
    return hasmethod(Base.size, Tuple{T}) ? Base.size(x)[1:(end - 1)] : nothing
end

@non_differentiable size_unbatched(::Any)

unbatched_structure(x) = fmapstructure(size_unbatched, x)

# Can we convert this to a NamedTuple?
can_named_tuple(::NamedTuple) = true
can_named_tuple(::T) where {T} = can_named_tuple(T)
function can_named_tuple(::Type{T}) where {T}
    return Core.Compiler._return_type(named_tuple, Tuple{T}) !== Union{}
end

@non_differentiable can_named_tuple(::Any)

# Convert to a NamedTuple
named_tuple(nt::NamedTuple) = nt
function named_tuple(x::T) where {T}
    NT = Core.Compiler._return_type(NamedTuple, Tuple{T})
    if NT === Union{} || NT === NamedTuple
        error("`NamedTuple` is not defined for type `$(T)`. Please define \
               `Lux.Utils.named_tuple(::$(T))` method (or preferably \
               `NamedTuple(::$(T))`).")
    end
    return NamedTuple(x)
end

# A more generalized version of `merge` that works with non-NamedTuples
merge(nt₁::NamedTuple, nt₂::NamedTuple) = Base.merge(nt₁, nt₂)
function merge(p, nt::NamedTuple)
    can_named_tuple(p) && return merge(named_tuple(p), nt)
    @argcheck length(p) == 0
    return nt
end
function merge(nt::NamedTuple, p)
    can_named_tuple(p) && return merge(nt, named_tuple(p))
    @argcheck length(p) == 0
    return nt
end
function merge(x, y)
    can_named_tuple(x) && return merge(named_tuple(x), y)
    can_named_tuple(y) && return merge(x, named_tuple(y))
    length(x) == 0 && return y
    length(y) == 0 && return x
    throw(ArgumentError(lazy"Cannot merge $(x)::$(typeof(x)) and $(y)::$(typeof(y)). Define `merge` method for these types."))
end

# Used in freezing
function pairs(x)
    can_named_tuple(x) && return Base.pairs(named_tuple(x))
    return Base.pairs(x)
end

@concrete struct Fix3 <: Function
    f
    x
end

Broadcast.broadcastable(f::Fix3) = Ref(f)

(f::Fix3)(a, b) = f.f(a, b, f.x)

# Take a `Val` and return the value. Noop for other types
unwrap_val(::Val{T}) where {T} = T
unwrap_val(x) = x

contiguous(x::AbstractArray) = x
contiguous(x::SubArray) = copy(x)

gate(h::Int, n::Int) = (1:h) .+ h * (n - 1)
gate(x::AbstractVector, h::Int, n::Int) = view(x, gate(h, n))
gate(x::AbstractMatrix, h::Int, n::Int) = view(x, gate(h, n), :)

reverse(x::AbstractArray; dims=:) = Base.reverse(x; dims)

vec(x::AbstractArray) = Base.vec(x)
vec(::Nothing) = nothing

function CRC.rrule(::typeof(vec), x::AbstractArray)
    return Base.vec(x), Δ -> (NoTangent(), reshape(CRC.unthunk(Δ), size(x)))
end

function sample_replicate(rng::AbstractRNG)
    rand(rng)
    return LuxCore.replicate(rng)
end

function index_namedtuple(nt::NamedTuple{fields}, idxs::AbstractArray) where {fields}
    return NamedTuple{fields[idxs]}(values(nt)[idxs])
end

eltype(x) = eltype(Base.eltype(x))
eltype(::Type{T}) where {T} = T
eltype(::Type{<:Dual{T, V}}) where {T, V} = V

@non_differentiable eltype(::Any)

ofeltype_array(::Type{T}, x::AbstractArray) where {T} = broadcast(T, x)
function ofeltype_array(::Type{T}, x::AbstractArray{<:Dual{Tag, V, N}}) where {Tag, T, V, N}
    return Dual{Tag, T, N}.(x)
end

function warn_mismatch(layer, x, warn_msg::AbstractString)
    @warn warn_msg layer summary(x) maxlog=1
end

@non_differentiable warn_mismatch(::Any, ::Any, ::Any)

zero(x) = Base.zero(x)
zero(::Nothing) = nothing
zero(x::Val) = x

zero!!(x::Number) = Base.zero(x)
function zero!!(x::AbstractArray{<:Number})
    fill!(x, false)
    return x
end
zero!!(::Nothing) = nothing
zero!!(x::Val) = x

function add!!(x::AbstractArray{<:Number}, y::AbstractArray{<:Number})
    ArrayInterface.can_setindex(x) || return x .+ y
    @. x += y
    return x
end
add!!(x::Number, y::Number) = x + y
add!!(::Nothing, ::Nothing) = nothing

function init_rnn_hidden_state(rng::AbstractRNG, rnn, x::AbstractMatrix)
    y = similar(x, rnn.out_dims, Base.size(x, 2))
    copyto!(y, rnn.init_state(rng, size(y)...))
    return y
end

@non_differentiable init_rnn_hidden_state(::Any...)

function init_trainable_rnn_hidden_state(hidden_state::AbstractVector, x::AbstractMatrix)
    return repeat(hidden_state, 1, Base.size(x, 2))
end

norm(x; dims=Colon()) = sqrt.(sum(abs2, x; dims))

function norm_except(x::AbstractArray{T, N}; dims::Union{Int, Tuple}=N) where {T, N}
    return norm(x; dims=get_norm_except_dims(ndims(x), dims))
end

get_norm_except_dims(N, dim::Int) = filter(i -> i != dim, 1:N)
get_norm_except_dims(N, dims::Tuple) = filter(i -> i ∉ dims, 1:N)

@non_differentiable get_norm_except_dims(::Any...)

expand(_, i::Tuple) = i
expand(N, i::Integer) = ntuple(Returns(i), N)

@non_differentiable expand(::Any...)

stack1(xs) = mapfoldl(expanddims1, vcat, xs)
expanddims1(x) = reshape(x, 1, size(x)...)

set_refval!(x, y) = (x[] = y)

@non_differentiable set_refval!(::Any...)
EnzymeRules.inactive(::typeof(set_refval!), ::Any...) = nothing

function named_tuple_layers(layers::Vararg{AbstractLuxLayer, N}) where {N}
    return NamedTuple{ntuple(i -> Symbol(:layer_, i), N)}(layers)
end

make_abstract_matrix(x::AbstractVector) = reshape(x, :, 1)
make_abstract_matrix(x::SVector{L, T}) where {L, T} = SMatrix{L, 1, T}(x)
make_abstract_matrix(x::AbstractMatrix) = x
make_abstract_matrix(x::AbstractArray{T, N}) where {T, N} = reshape(x, Base.size(x, 1), :)

matrix_to_array(x::AbstractMatrix, ::AbstractVector) = vec(x)
matrix_to_array(x::SMatrix{L, 1, T}, ::AbstractVector) where {L, T} = SVector{L, T}(x)
matrix_to_array(x::AbstractMatrix, ::AbstractMatrix) = x
matrix_to_array(x::AbstractMatrix, y::AbstractArray) = reshape(x, :, size(y)[2:end]...)

# This should probably be in WeightInitializers.jl
calculate_gain(_, __) = 1.0f0
calculate_gain(::typeof(identity), _) = 1.0f0
calculate_gain(::typeof(NNlib.sigmoid), _) = 1.0f0
calculate_gain(::typeof(NNlib.sigmoid_fast), _) = 1.0f0
calculate_gain(::typeof(NNlib.relu), _) = 2.0f0
calculate_gain(::typeof(tanh), _) = 5.0f0 / 3.0f0
calculate_gain(::typeof(NNlib.tanh_fast), _) = 5.0f0 / 3.0f0
function calculate_gain(::typeof(NNlib.leakyrelu), ::Nothing)
    return calculate_gain(NNlib.leakyrelu, 0.1f0)
end
calculate_gain(::typeof(NNlib.leakyrelu), x::Real) = typeof(x)(√(2 / (1 + x^2)))
calculate_gain(::typeof(NNlib.selu), _) = 3.0f0 / 4

end

using .Utils: Utils, BoolType, IntegerType, SymbolType, make_abstract_matrix,
              matrix_to_array, init_trainable_rnn_hidden_state, init_rnn_hidden_state

const safe_reverse = Utils.reverse
const safe_vec = Utils.vec
const unwrapped_eltype = Utils.eltype
