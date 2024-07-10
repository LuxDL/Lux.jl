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

# Val utilities
__unwrap_val(::Val{T}) where {T} = T

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

# Merging Exotic Types
_merge(nt1::NamedTuple, nt2::NamedTuple) = merge(nt1, nt2)
function _merge(p, nt::NamedTuple)
    __can_named_tuple(p) && return _merge(__named_tuple(p), nt)
    @argcheck length(p) == 0
    return nt
end
function _merge(nt::NamedTuple, p)
    __can_named_tuple(p) && return _merge(nt, __named_tuple(p))
    @argcheck length(p) == 0
    return nt
end
function _merge(x, y)
    __can_named_tuple(x) && return _merge(__named_tuple(x), y)
    __can_named_tuple(y) && return _merge(x, __named_tuple(y))
    length(x) == 0 && return y
    length(y) == 0 && return x
    throw(ArgumentError(lazy"Cannot merge $(x)::$(typeof(x)) and $(y)::$(typeof(y)). Define `_merge` method for these types."))
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
@inline __stack1(xs) = mapfoldl(__expanddims1, vcat, xs)

@inline __expanddims1(x) = reshape(x, 1, size(x)...)

# Used in freezing
## Extend for custom types
@inline function _pairs(x)
    __can_named_tuple(x) && return pairs(__named_tuple(x))
    return pairs(x)
end

@inline __named_tuple(nt::NamedTuple) = nt
@inline function __named_tuple(x::T) where {T}
    NT = Core.Compiler._return_type(NamedTuple, Tuple{T})
    if NT === Union{} || NT === NamedTuple
        error("`NamedTuple` is not defined for type `$(T)`. Please define \
               `Lux.__named_tuple(::$(T))` method (or preferably `NamedTuple(::$(T))`).")
    end
    return NamedTuple(x)
end

@inline __can_named_tuple(::NamedTuple) = true
@inline __can_named_tuple(::T) where {T} = __can_named_tuple(T)
@inline function __can_named_tuple(::Type{T}) where {T}
    return Core.Compiler._return_type(__named_tuple, Tuple{T}) !== Union{}
end

@inline _vec(x::AbstractArray) = vec(x)
@inline _vec(::Nothing) = nothing

@inline function __named_tuple_layers(layers::Vararg{AbstractExplicitLayer, N}) where {N}
    return NamedTuple{ntuple(i -> Symbol(:layer_, i), N)}(layers)
end

@inline __size(x::AbstractArray) = size(x)
@inline __size(x::T) where {T} = hasmethod(size, Tuple{T}) ? size(x) : nothing

# helpers for the loss functions
"""
    xlogx(x::Number)

Return `x * log(x)` for `x ≥ 0`, handling `x == 0` by taking the limit from above, to get
zero.
"""
@inline function xlogx(x::Number)
    result = x * log(x)
    return ifelse(iszero(x), zero(result), result)
end

"""
    xlogy(x::Number, y::Number)

Return `x * log(y)` for `y > 0`, and zero when `x == 0`.
"""
@inline function xlogy(x::Number, y::Number)
    result = x * log(y)
    return ifelse(iszero(x), zero(result), result)
end

# Some functional forms of losses
@concrete struct __Fix3
    f
    x
end

Broadcast.broadcastable(f::__Fix3) = Ref(f)

@inline (f::__Fix3)(a, b) = f.f(a, b, f.x)

@inline function __siamese_contrastive_loss(x::T1, y::T2, margin=true) where {T1, T2}
    return (1 - y) * x^2 + y * max(promote_type(T1, T2)(0), margin - x)^2
end

@inline function __poisson_loss(x::T1, y::T2, ϵ) where {T1, T2}
    return x - xlogy(y, x + __get_epsilon(T1, ϵ))
end

@inline function __msle_loss(x::T1, y::T2, ϵ) where {T1, T2}
    ϵ = __get_epsilon(promote_type(T1, T2), ϵ)
    return log((x + ϵ) / (y + ϵ))^2
end

# Misc Utils
@inline function __check_sizes(ŷ::AbstractArray, y::AbstractArray)
    for d in 1:max(ndims(ŷ), ndims(y))
        if size(ŷ, d) != size(y, d)
            throw(DimensionMismatch("loss function expects size(ŷ) = $(size(ŷ)) to match \
                                     size(y) = $(size(y))"))
        end
    end
end
@inline __check_sizes(ŷ, y) = nothing

@inline function __fused_agg(::typeof(mean), op::OP, x) where {OP}
    return __fused_agg(sum, op, x) / length(x)
end
@inline function __fused_agg(::typeof(mean), lfn::LossFunctions.Traits.Loss, x, y)
    return __fused_agg(sum, lfn, x, y) / length(x)
end

@inline __fused_agg(::typeof(sum), op::OP, x::Number) where {OP} = op(x)
@inline __fused_agg(::typeof(sum), op::OP, x) where {OP} = sum(op, x)
@inline function __fused_agg(
        ::typeof(sum), lfn::LossFunctions.Traits.Loss, x::Number, y::Number)
    return lfn(x, y)
end
@inline function __fused_agg(::typeof(sum), lfn::LossFunctions.Traits.Loss, x, y)
    fast_scalar_indexing(x) && fast_scalar_indexing(y) && return sum(lfn, x, y)
    # mapreduce(Broadcast.BroadcastFunction(lfn), +, x, y) leads to slowdowns, better to
    # allocate a new array
    return sum(lfn.(x, y))
end

@inline __fused_agg(::Nothing, op::OP, args...) where {OP} = op.(args...)
@inline __fused_agg(f::F, op::OP, args...) where {F, OP} = f(op.(args...))

@inline __label_smoothing(::Nothing, y, ::Type{T}) where {T} = y
@inline function __label_smoothing(label_smoothing::Real, y, ::Type{T}) where {T}
    label_smoothing = T(label_smoothing)
    return y .* (1 - label_smoothing) .+ label_smoothing ./ size(y, ndims(y) - 1)
end

@inline __label_smoothing_binary(::Nothing, y, ::Type{T}) where {T} = y
@inline function __label_smoothing_binary(label_smoothing::Real, y, ::Type{T}) where {T}
    label_smoothing = T(label_smoothing)
    return y .* (1 - label_smoothing) .+ label_smoothing ./ 2
end

@inline __get_epsilon(::Type{T}, ϵ::Real) where {T} = T(ϵ)
@inline __get_epsilon(::Type{T}, ::Nothing) where {T} = eps(float(T))

@inline __get_dims(::AbstractVector) = Colon()
@inline __get_dims(::AbstractArray{T, N}) where {T, N} = 1:(N - 1)

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

@inline __reverse(x; dims=:) = reverse(x; dims)
