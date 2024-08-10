module Utils

using ArgCheck: @argcheck
using ChainRulesCore: @non_differentiable
using ConcreteStructs: @concrete
using Functors: fmapstructure

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
@inline named_tuple(nt::NamedTuple) = nt
@inline function named_tuple(x::T) where {T}
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
@inline function pairs(x)
    can_named_tuple(x) && return Base.pairs(Utils.named_tuple(x))
    return Base.pairs(x)
end

@concrete struct Fix3
    f
    x
end

Broadcast.broadcastable(f::Fix3) = Ref(f)

(f::Fix3)(a, b) = f.f(a, b, f.x)

unwrap_val(::Val{T}) where {T} = T

end

# Training Check
"""
    istraining(::Val{training})
    istraining(::Bool)
    istraining(st::NamedTuple)

Returns `true` if `training` is `true` or if `st` contains a `training` field with value
`true`. Else returns `false`.
"""
@inline istraining(::Val{training}) where {training} = training
@inline istraining(training::Bool) = training
@inline istraining(st::NamedTuple) = hasproperty(st, :training) && istraining(st.training)

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

@inline __tuple_string(pad::Tuple) = all(==(pad[1]), pad) ? string(pad[1]) : string(pad)

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

# RNN Utilities
@inline _gate(h::Int, n::Int) = (1:h) .+ h * (n - 1)
@inline _gate(x::AbstractVector, h::Int, n::Int) = view(x, _gate(h, n))
@inline _gate(x::AbstractMatrix, h::Int, n::Int) = view(x, _gate(h, n), :)

@inline function _init_hidden_state(rng::AbstractRNG, rnn, x::AbstractMatrix)
    return rnn.init_state(rng, rnn.out_dims, size(x, 2)) |> get_device(x)
end

@inline function _init_trainable_hidden_state(
        hidden_state::AbstractVector, x::AbstractMatrix)
    return repeat(hidden_state, 1, size(x, 2))
end

"""
    multigate(x::AbstractArray, ::Val{N})

Split up `x` into `N` equally sized chunks (along dimension `1`).
"""
@inline function multigate(x::AbstractArray, ::Val{N}) where {N}
    return ntuple(i -> _gate(x, size(x, 1) ÷ N, i), N)
end

# Indexing into NamedTuple
function _index_namedtuple(nt::NamedTuple{fields}, idxs::AbstractArray) where {fields}
    return NamedTuple{fields[idxs]}(values(nt)[idxs])
end

# If doesn't have a property, return nothing
@generated function _getproperty(x::NamedTuple{names}, ::Val{v}) where {names, v}
    return v ∈ names ? :(x.$v) : :(nothing)
end

## Slow-fallback
@inline function _getproperty(x, ::Val{v}) where {v}
    return v ∈ propertynames(x) ? getproperty(x, v) : nothing
end

@inline function _eachslice(x::AbstractArray, ::Val{dims}) where {dims}
    return [selectdim(x, dims, i) for i in axes(x, dims)]
end
@inline function _eachslice(x::GPUArraysCore.AnyGPUArray, ::Val{dims}) where {dims}
    return [__unview(selectdim(x, dims, i)) for i in axes(x, dims)]
end

@inline __unview(x::SubArray) = copy(x)
@inline __unview(x) = x

function ∇_eachslice(Δ_raw, x::AbstractArray, ::Val{dims}) where {dims}
    Δs = CRC.unthunk(Δ_raw)
    i1 = findfirst(Δ -> Δ isa AbstractArray, Δs)
    i1 === nothing && zero.(x)  # all slices are Zero!
    Δ = similar(x)
    for i in axes(x, dims)
        Δi = selectdim(Δ, dims, i)
        copyto!(Δi, Δs[i])
    end
    return CRC.ProjectTo(x)(Δ)
end

# Backend Integration
## Convolution
@inline _conv_transpose(x, weight, cdims) = LuxLib.__∇conv_data(x, weight, cdims)

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
@inline function compute_adaptive_pooling_dims(x::AbstractArray, outsize)
    insize = size(x)[1:(end - 2)]
    stride = insize .÷ outsize
    k = insize .- (outsize .- 1) .* stride
    pad = 0
    return PoolDims(x, k; padding=pad, stride=stride)
end

## Foldl with init
"""
    foldl_init(op, x)
    foldl_init(op, x, init)

Exactly same as `foldl(op, x; init)` in the forward pass. But, gives gradients wrt `init`
in the backward pass.
"""
@inline foldl_init(op, x) = foldl_init(op, x, nothing)
@inline foldl_init(op, x, init) = foldl(op, x; init)

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
@inline __stack1(xs) = mapfoldl(__expanddims1, vcat, xs)

@inline __expanddims1(x) = reshape(x, 1, size(x)...)

@inline _vec(x::AbstractArray) = vec(x)
@inline _vec(::Nothing) = nothing

@inline function __named_tuple_layers(layers::Vararg{AbstractExplicitLayer, N}) where {N}
    return NamedTuple{ntuple(i -> Symbol(:layer_, i), N)}(layers)
end

@inline __zero(x) = zero(x)
@inline __zero(::Nothing) = nothing
@inline __zero(x::Val) = x

@inline __zero!!(x::Number) = zero(x)
@inline __zero!!(x::AbstractArray{<:Number}) = fill!(x, zero(eltype(x)))
@inline __zero!!(::Nothing) = nothing
@inline __zero!!(x::Val) = x

@inline function __add!!(x::AbstractArray{<:Number}, y::AbstractArray{<:Number})
    ArrayInterface.can_setindex(x) || return x .+ y
    @. x += y
    return x
end
@inline __add!!(x::Number, y::Number) = x + y
@inline __add!!(::Nothing, ::Nothing) = nothing

@inline __set_refval!(x, y) = (x[] = y)

@inline __eltype(x) = eltype(x)
@inline __eltype(::ForwardDiff.Dual{T, V}) where {T, V} = V
@inline __eltype(::AbstractArray{<:ForwardDiff.Dual{T, V}}) where {T, V} = V

@inline __reverse(x; dims=:) = reverse(x; dims)
