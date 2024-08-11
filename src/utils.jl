module Utils

using ArgCheck: @argcheck
using ChainRulesCore: @non_differentiable
using ConcreteStructs: @concrete
using ForwardDiff: Dual
using Functors: fmapstructure
using Random: AbstractRNG

using LuxCore: LuxCore

# Aliased `size` from Base
size(x::AbstractArray) = Base.size(x)
size(x::T) where {T} = hasmethod(Base.size, Tuple{T}) ? Base.size(x) : nothing

@non_differentiable size(::Any)

structure(x) = fmapstructure(size, x)

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
    can_named_tuple(x) && return Base.pairs(Utils.named_tuple(x))
    return Base.pairs(x)
end

@concrete struct Fix3
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

ofeltype_array(::Type{T}, x::AbstractArray) where {T} = broadcast(T, x)
function ofeltype_array(::Type{T}, x::AbstractArray{<:Dual{Tag, V, N}}) where {Tag, T, V, N}
    return Dual{Tag, T, N}.(x)
end

function warn_mismatch(layer, x, warn_msg::AbstractString)
    @warn warn_msg layer summary(x) maxlog=1
end

@non_differentiable warn_mismatch(::Any, ::Any, ::Any)

end

# Convolution
function _convfilter(rng::AbstractRNG, filter::NTuple{N, Integer},
        ch::Pair{<:Integer, <:Integer}; init=glorot_uniform, groups=1) where {N}
    cin, cout = ch
    @argcheck cin % groups==0 DimensionMismatch("Input channel dimension must be divisible by groups.")
    @argcheck cout % groups==0 DimensionMismatch("Output channel dimension must be divisible by groups.")
    return init(rng, filter..., cin ÷ groups, cout)
end

_expand(N, i::Tuple) = i
_expand(N, i::Integer) = ntuple(_ -> i, N)

__tuple_string(pad::Tuple) = all(==(pad[1]), pad) ? string(pad[1]) : string(pad)

# Padding
struct SamePad end

function _calc_padding(pad, ::NTuple{N}, dilation, stride) where {N}
    return _expand(Val(2 * N), pad)
end

function _calc_padding(::SamePad, k::NTuple, dilation, stride)
    # Ref: "A guide to convolution arithmetic for deep learning"
    # https://arxiv.org/abs/1603.07285 Effective kernel size, including dilation
    k_eff = @. k + (k - 1) * (dilation - 1)
    # How much total padding needs to be applied?
    pad_amt = @. k_eff - 1
    # In case amount of padding is odd we need to apply different amounts to each side.
    return Tuple(mapfoldl(i -> [cld(i, 2), fld(i, 2)], vcat, pad_amt))
end

function _init_hidden_state(rng::AbstractRNG, rnn, x::AbstractMatrix)
    return rnn.init_state(rng, rnn.out_dims, size(x, 2)) |> get_device(x)
end

function _init_trainable_hidden_state(hidden_state::AbstractVector, x::AbstractMatrix)
    return repeat(hidden_state, 1, size(x, 2))
end

# Backend Integration
## Convolution
_conv_transpose(x, weight, cdims) = LuxLib.__∇conv_data(x, weight, cdims)

function _conv_transpose_dims(
        x::AbstractArray, weight::AbstractArray; padding, stride, dilation, groups)
    # Calculate size of "input", from ∇conv_data()'s perspective...
    function calc_dim(xsz, wsz, stride, dilation, pad)
        return (xsz - 1) * stride + 1 + (wsz - 1) * dilation - pad
    end
    combined_pad = ntuple(i -> padding[2i - 1] + padding[2i], length(padding) ÷ 2)
    I = map(calc_dim, size(x)[1:(end - 2)], size(weight)[1:(end - 2)],
        stride, dilation, combined_pad)
    C_in = size(weight)[end - 1] * groups
    C_out = size(weight)[end]
    batch_size = size(x)[end]
    w_size = size(weight)

    size(x)[end - 1] != C_out &&
        throw(DimensionMismatch(lazy"Expected $(C_out) input channels but got $(size(x)[end - 1]) channels."))

    # Create DenseConvDims() that looks like the corresponding conv()
    return DenseConvDims(
        (I..., C_in, batch_size), w_size; stride, padding, dilation, groups)
end

## Adaptive Pooling
function compute_adaptive_pooling_dims(x::AbstractArray, outsize)
    insize = size(x)[1:(end - 2)]
    stride = insize .÷ outsize
    k = insize .- (outsize .- 1) .* stride
    pad = 0
    return PoolDims(x, k; padding=pad, stride=stride)
end

# Utility Functions to Convert Parameter and State Types
struct LuxEltypeAdaptor{T} end

(l::LuxEltypeAdaptor)(x) = fmap(adapt(l), x)
function (l::LuxEltypeAdaptor)(x::AbstractArray{T}) where {T}
    return isbitstype(T) ? adapt(l, x) : map(adapt(l), x)
end

function Adapt.adapt_storage(
        ::LuxEltypeAdaptor{T}, x::AbstractArray{<:AbstractFloat}) where {T <: AbstractFloat}
    return convert(AbstractArray{T}, x)
end

function Adapt.adapt_storage(::LuxEltypeAdaptor{T},
        x::AbstractArray{<:Complex{<:AbstractFloat}}) where {T <: AbstractFloat}
    return convert(AbstractArray{Complex{T}}, x)
end

for (fname, ftype) in zip((:f16, :f32, :f64), (Float16, Float32, Float64))
    @eval begin
        """
            $($fname)(m)

        Converts the `eltype` of `m` *floating point* values to `$($ftype)`.
        Recurses into structs marked with `Functors.@functor`.
        """
        $(fname)(m) = (LuxEltypeAdaptor{$ftype}())(m)
    end
end

# Common incorrect usage
for f in (f16, f32, f64)
    warn_msg = lazy"$(f) is not meant to be broadcasted like `$(f).(x)` or `x .|> $(f)`, and this might give unexpected results and could lead to crashes. Directly use `$(f)` as `$(f)(x)` or `x |> $(f)` instead."
    @eval begin
        function Base.Broadcast.broadcasted(::typeof($(f)), arg1)
            @warn $(warn_msg)
            arg1′ = Broadcast.broadcastable(arg1)
            return Broadcast.broadcasted(Broadcast.combine_styles(arg1′), $(f), arg1′)
        end

        function Base.Broadcast.broadcasted(::typeof(|>), arg1, ::typeof($(f)))
            @warn $(warn_msg)
            arg1′ = Broadcast.broadcastable(arg1)
            return Broadcast.broadcasted(Broadcast.combine_styles(arg1′), $(f), arg1′)
        end
    end
end

# Stack using vcat and mapreduce
# FIXME: Just use `stack` once the rrule is fixed upstream
__stack1(xs) = mapfoldl(__expanddims1, vcat, xs)

__expanddims1(x) = reshape(x, 1, size(x)...)

function __named_tuple_layers(layers::Vararg{AbstractExplicitLayer, N}) where {N}
    return NamedTuple{ntuple(i -> Symbol(:layer_, i), N)}(layers)
end

__zero(x) = zero(x)
__zero(::Nothing) = nothing
__zero(x::Val) = x

__zero!!(x::Number) = zero(x)
__zero!!(x::AbstractArray{<:Number}) = fill!(x, zero(eltype(x)))
__zero!!(::Nothing) = nothing
__zero!!(x::Val) = x

function __add!!(x::AbstractArray{<:Number}, y::AbstractArray{<:Number})
    ArrayInterface.can_setindex(x) || return x .+ y
    @. x += y
    return x
end
__add!!(x::Number, y::Number) = x + y
__add!!(::Nothing, ::Nothing) = nothing

__set_refval!(x, y) = (x[] = y)
