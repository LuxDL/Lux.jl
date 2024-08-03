# Various Array Traits
function fast_scalar_indexing(::T) where {T <: AbstractArray}
    return static(ArrayInterface.fast_scalar_indexing(T))
end
fast_scalar_indexing(::Nothing) = True()

is_mutable_array(::T) where {T <: AbstractArray} = static(can_setindex(T))
is_mutable_array(::Nothing) = True()

for op in (:has_dual, :has_float16, :is_tracked)
    @eval $op(::Nothing) = False()
    @eval $op(x::Numeric) = $op(eltype(x))
end

has_dual(::Type{<:Number}) = False()
has_dual(::Type{<:ForwardDiff.Dual}) = True()

has_float16(::Type{<:Number}) = False()
has_float16(::Type{<:Float16}) = True()

is_tracked(::Type{<:Number}) = False()

has_autodiff_value(x) = is_tracked(x) | has_dual(x)

static_isa(::Type{T}) where {T} = Base.Fix2(static_isa, T)
static_isa(x, ::Type{T}) where {T} = static(isa(x, T))

# Current Checks. If any of these are false, we fallback to the generic implementation.
#   - Is Mutable
#   - Doesn't Has Dual Numbers
attempt_fast_implementation(x) = attempt_fast_implementation((x,))
function attempt_fast_implementation(xs::Tuple)
    return unrolled_all(is_mutable_array, xs) & unrolled_all(!has_dual, xs)
end

CRC.@non_differentiable attempt_fast_implementation(::Any...)
EnzymeRules.inactive_noinl(::typeof(attempt_fast_implementation), ::Any...) = nothing

function use_generic_broadcasting(xs::Tuple)
    # Float16 is a bit iffy and reordering operations are not optimal for numerical
    # stability so we use the generic implementation for now.
    return unrolled_any(has_autodiff_value, xs) |
           unrolled_any(has_float16, xs) |
           unrolled_any(static_isa(StaticArray), xs)
end

# How to do an internal operation?
#    1. Generic Broadcasting without Preallocation -- GenericBroadcastOp
#    2. Broadcasting with Fusion -- GPUBroadcastOp
#    3. Use Loops possibly accelerating with LoopVectorization or Polyester. This might
#       still use broadcasting if needed

abstract type AbstractInternalArrayOpMode end

abstract type AbstractBroadcastOpMode <: AbstractInternalArrayOpMode end

struct GenericBroadcastOp <: AbstractBroadcastOpMode end
struct GPUBroadcastOp{dev} <: AbstractBroadcastOpMode end
struct LoopedArrayOp <: AbstractInternalArrayOpMode end

## NOTE: Ensure that this always gets compiled out! Else we will have terrible type
##       inference.
function internal_operation_mode(xs::Tuple)
    xs = unrolled_filter(!isnothing, xs)
    known(use_generic_broadcasting(xs)) && return GenericBroadcastOp()

    dev = get_device_type(xs)
    dev <: AbstractGPUDevice && return GPUBroadcastOp{dev}()

    # This check needs to be done after the GPU Check
    known(unrolled_any(!fast_scalar_indexing, xs)) && return GenericBroadcastOp()
    return LoopedArrayOp()
end
internal_operation_mode(x::AbstractArray) = internal_operation_mode((x,))

CRC.@non_differentiable internal_operation_mode(::Any...)
EnzymeRules.inactive_noinl(::typeof(internal_operation_mode), ::Any...) = nothing
