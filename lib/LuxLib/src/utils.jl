const Optional{T} = Union{Nothing, T}
const Numeric = Union{AbstractArray{<:T}, T} where {T <: Number}
const ∂∅ = NoTangent()

# Bias Gradient -- can't be used inside gradient rules
__added_bias_gradient(::Nothing, Δ::AbstractArray) = ∂∅
function __added_bias_gradient(
        b::AbstractArray{<:Number, N}, Δ::AbstractArray{<:Number, N}) where {N}
    return __reduce_sum(b, Δ)
end
function __added_bias_gradient(b::AbstractVector{<:Number}, Δ::AbstractArray{<:Number})
    b_ = __reshape_bias_into_xdims(Δ, b)
    return vec(__reduce_sum(b_, Δ))
end

# Operations that most AD won't be able to differentiate
__reduce_sum(::Nothing, ::NoTangent) = ∂∅
function __reduce_sum(x::AbstractArray, y::AbstractArray)
    z = similar(x, promote_type(eltype(x), eltype(y)))
    sum!(z, y)
    return z
end

# Simple Operations -- no rrules needed
@generated _vec(x::T) where {T} = hasmethod(vec, (T,)) ? :(vec(x)) : :x

## Maybe typecast the array
_ofeltype_array(::Type{T}, x::AbstractArray{T}) where {T} = x
_ofeltype_array(::Type{T}, x::AbstractArray) where {T} = convert(AbstractArray{T}, x)
_ofeltype_array(::Type{T}, ::Nothing) where {T} = nothing

__materialize_subarray(x::AbstractArray) = x
__materialize_subarray(x::SubArray) = copy(x)

__value(x::Number) = x
__value(x::AbstractArray) = x
__value(::Type{T}) where {T <: Number} = T
__value(x::ForwardDiff.Dual) = ForwardDiff.value(x)
__value(x::AbstractArray{<:ForwardDiff.Dual}) = ForwardDiff.value.(x)
__value(::Type{<:ForwardDiff.Dual{Tag, T}}) where {Tag, T} = __value(T)
__value(::Nothing) = nothing

__aos_to_soa(x::AbstractArray) = x # FIXME: Upstream this to ArrayInterface.jl

__reshape(x::AbstractArray, dims...) = reshape(x, dims)
__reshape(::Nothing, dims...) = nothing

# Non-differentiable functions
## Reduce BLAS threads if we are going to use a native Julia implementation
function __maybe_reduce_BLAS_threads(x::AbstractArray)
    __maybe_reduce_BLAS_threads(get_device_type(x))
end
__maybe_reduce_BLAS_threads(::Type{T}) where {T} = -1
function __maybe_reduce_BLAS_threads(::Type{LuxCPUDevice})::Int
    old_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    return old_threads
end

CRC.@non_differentiable __maybe_reduce_BLAS_threads(::AbstractArray)
EnzymeRules.inactive_noinl(::typeof(__maybe_reduce_BLAS_threads), ::AbstractArray) = nothing

function __reset_BLAS_threads(old_threads::Int)
    old_threads ≥ 1 && BLAS.set_num_threads(old_threads)
    return nothing
end

CRC.@non_differentiable __reset_BLAS_threads(::Int)
EnzymeRules.inactive_noinl(::typeof(__reset_BLAS_threads), ::Int) = nothing

## Check no setindexing
__is_immutable_array(x::AbstractArray) = !can_setindex(x)
__is_immutable_array(::Nothing) = false
__is_immutable_array_val(x) = Val(__is_immutable_array(x))

CRC.@non_differentiable __is_immutable_array_val(::Any...)
EnzymeRules.inactive_noinl(::typeof(__is_immutable_array_val), ::Any...) = nothing

__has_dual(x) = false
__has_dual(::ForwardDiff.Dual) = true
__has_dual(::AbstractArray{<:ForwardDiff.Dual}) = true

__is_immutable_array_or_dual(x) = __is_immutable_array(x) || __has_dual(x)
function __is_immutable_array_or_dual_val(x::Tuple)
    return Val(unrolled_any(__is_immutable_array_or_dual, x))
end

CRC.@non_differentiable __is_immutable_array_or_dual_val(::Any...)
EnzymeRules.inactive_noinl(::typeof(__is_immutable_array_or_dual_val), ::Any...) = nothing

function __get_concrete_fba_output_eltype(act::F, ::AbstractArray{Tw}, ::AbstractArray{Tx},
        b::Optional{<:AbstractVector}) where {F, Tw, Tx}
    if b === nothing
        Ty = promote_type(Tw, Tx)
        Tact = Core.Compiler._return_type(act, Tuple{Ty})
        return ifelse(isconcretetype(Tact), Tact, Ty)
    end
    Ty = promote_type(Tw, Tx, eltype(b))
    Tact = Core.Compiler._return_type(act, Tuple{Ty})
    return ifelse(isconcretetype(Tact), Tact, Ty)
end

function __get_concrete_fba_output_eltype(
        act::F, x::AbstractArray, b::Optional{<:AbstractVector}) where {F}
    return __get_concrete_fba_output_eltype(act, x, x, b)
end

CRC.@non_differentiable __get_concrete_fba_output_eltype(::Any...)
EnzymeRules.inactive_noinl(::typeof(__get_concrete_fba_output_eltype), ::Any...) = nothing

## Copy and don't allow gradient propagation
_copy_autodiff_barrier(x) = copy(__value(x))
_copy_autodiff_barrier(::Nothing) = nothing

CRC.@non_differentiable _copy_autodiff_barrier(::Any)
EnzymeRules.inactive_noinl(::typeof(_copy_autodiff_barrier), ::Any...) = nothing

__has_tracked_value(::Any) = false

CRC.@non_differentiable __has_tracked_value(::Any)
EnzymeRules.inactive_noinl(::typeof(__has_tracked_value), ::Any) = nothing

__has_autodiff_value(x) = __has_tracked_value(x) || __has_dual(x)

## depwarn but marked non-differentiable to prevent type instability
__depwarn(msg::String, f::Symbol) = Base.depwarn(msg, f)

CRC.@non_differentiable __depwarn(::Any...)
EnzymeRules.inactive_noinl(::typeof(__depwarn), ::Any...) = nothing

__eltype(::AbstractArray{T}) where {T} = T
__eltype(::Nothing) = Bool

CRC.@non_differentiable __eltype(::Any)
EnzymeRules.inactive_noinl(::typeof(__eltype), ::Any) = nothing

__has_float16(::Type{T}) where {T} = T <: Float16
__has_float16(::AbstractArray{T}) where {T} = __has_float16(T)
__has_float16(::Float16) = true
__has_float16(x) = false

CRC.@non_differentiable __has_float16(::Any)
EnzymeRules.inactive_noinl(::typeof(__has_float16), ::Any) = nothing

__default_epsilon(::Type{T}) where {T} = T(eps(T)^(5 / 7))
__default_epsilon(::AbstractArray{T}) where {T} = __default_epsilon(T)

CRC.@non_differentiable __default_epsilon(::Any...)
EnzymeRules.inactive_noinl(::typeof(__default_epsilon), ::Any...) = nothing

# Meta Programming Utilities
__is_tracked(x) = x == :TrackedArray || x == :TrackedVector
__is_tracked(args...) = any(__is_tracked, args)

## This part is taken from NNlib.jl
# This just saves typing `only.(only.(` many times:
only_derivative(y, f::F, x) where {F} = only(only(CRC.derivatives_given_output(y, f, x)))

# This has no methods, used for testing whether `derivatives_given_output(Ω, f, x)`
# is independent of `x`, as `_return_type` says `Union{}` when calling is an error.
struct NotaNumber <: Real end

# How to take activation gradients?
# See https://github.com/FluxML/NNlib.jl/blob/d85402aa39ddc6386d194e0dad88ab2e514ec5ea/src/bias_act.jl#L59-L60
function __no_intermediate_needed(f::F, ::Type{T}) where {F, T}
    f === identity && return true
    return isconcretetype(Core.Compiler._return_type(
        only_derivative, Tuple{T, F, NotaNumber}))
end

function __needs_intermediate_but_has_rrule(f::F, ::Type{T}) where {F, T}
    return isconcretetype(Core.Compiler._return_type(only_derivative, Tuple{T, F, T}))
end

# How to do a broadcast?
#    1. Generic Broadcasting without Preallocation -- GenericBroadcastOp
#    2. Broadcasting with Fusion -- GPUBroadcastOp
#    3. Loop Broadcasting -- LoopedArrayOp. This might still use broadcasting if needed

abstract type AbstractInternalArrayOpMode end

abstract type AbstractBroadcastOpMode <: AbstractInternalArrayOpMode end

struct GenericBroadcastOp <: AbstractBroadcastOpMode end
struct GPUBroadcastOp{dev} <: AbstractBroadcastOpMode end
struct LoopedArrayOp <: AbstractInternalArrayOpMode end

## NOTE: Ensure that this always gets compiled out! Else we will have terrible type
##       inference.
function internal_operation_mode(xs::Tuple)
    xs = unrolled_filter(!isnothing, xs)
    # Float16 is a bit iffy and reordering operations are not optimal for numerical
    # stability so we use the generic implementation for now.
    if unrolled_any(__has_autodiff_value, xs) || unrolled_any(__has_float16, xs)
        return GenericBroadcastOp()
    end
    dev = get_device_type(xs)
    dev <: AbstractLuxGPUDevice && return GPUBroadcastOp{dev}()
    unrolled_any(!fast_scalar_indexing, xs) && return GenericBroadcastOp()
    dev <: LuxCPUDevice && return LoopedArrayOp()
    return GenericBroadcastOp()  # fallback for safety
end
internal_operation_mode(x::AbstractArray) = internal_operation_mode((x,))

CRC.@non_differentiable internal_operation_mode(::Any...)
EnzymeRules.inactive_noinl(::typeof(internal_operation_mode), ::Any...) = nothing
