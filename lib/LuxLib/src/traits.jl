module Traits

using ArrayInterface: ArrayInterface, can_setindex
using ChainRulesCore: ChainRulesCore
using ForwardDiff: ForwardDiff
using NNlib: NNlib
using Static: True, False, static
using StaticArraysCore: StaticArray

using ..LuxLib: Numeric
using ..Utils

function fast_scalar_indexing(::T) where {T <: AbstractArray}
    return static(ArrayInterface.fast_scalar_indexing(T))
end
fast_scalar_indexing(::Nothing) = True()
fast_scalar_indexing(x::NNlib.BatchedAdjOrTrans) = fast_scalar_indexing(parent(x))

is_mutable_array(::T) where {T <: AbstractArray} = static(can_setindex(T))
is_mutable_array(::Nothing) = True()

ChainRulesCore.@non_differentiable is_mutable_array(::Any...)

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

function use_generic_broadcasting(xs::Tuple)
    # Float16 is a bit iffy and reordering operations are not optimal for numerical
    # stability so we use the generic implementation for now.
    return Utils.unrolled_any(has_autodiff_value, xs) |
           Utils.unrolled_any(has_float16, xs) |
           Utils.unrolled_any(static_isa(StaticArray), xs)
end

activation_intermediate_not_needed(::typeof(identity), ::Type) = True()

function activation_intermediate_not_needed(::F, ::Type{T}) where {F, T}
    return static(isconcretetype(Core.Compiler._return_type(
        Utils.only_derivative, Tuple{T, F, Utils.NotaNumber})))
end

function activation_has_rrule(::F, ::Type{T}) where {F, T}
    return static(isconcretetype(Core.Compiler._return_type(
        Utils.only_derivative, Tuple{T, F, T})))
end

end

module System

using ChainRulesCore: ChainRulesCore
using Static: True, False

using ..Utils

const CRC = ChainRulesCore

function explicit_blas_loaded()
    return Utils.is_extension_loaded(Val(:MKL)) |
           Utils.is_extension_loaded(Val(:AppleAccelerate)) |
           Utils.is_extension_loaded(Val(:BLISBLAS))
end

CRC.@non_differentiable explicit_blas_loaded()

function use_octavian()
    @static if Sys.ARCH == :x86_64  # Mostly from benchmarking we reach this point
        return !explicit_blas_loaded()
    else
        return False()
    end
end

CRC.@non_differentiable use_octavian()

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
"""
    internal_operation_mode(xs::Tuple)
    internal_operation_mode(x::AbstractArray)

Returns the internal operation mode for the given array(s). This is useful to define custom
implementations using different backends like simple Julia broadcasting, Kernel
Abstractions, Loop Vectorization, etc.

Currently supported modes are:

  - `GenericBroadcastOp`: This is the fallback for most types. For the following types this
    is the preferred mode:

      + Arrays with `fast_scalar_indexing` set to `False`.
      + Static Arrays
      + ReverseDiff Arrays
      + Tracker Arrays
      + ForwardDiff.Dual Arrays

  - `GPUBroadcastOp{dev}`: GPU Arrays where `dev` is obtained from `get_device_type(xs)`.
    This option dispatches should preferably use `KernelAbstractions` or specialized vendor
    dispatches.
  - `LoopedArrayOp`: CPU arrays that can be optimized using SIMD Loops, ideally using
    `LoopVectorization.jl` or `Polyester.jl`.
"""
function internal_operation_mode(xs::Tuple)
    xs = unrolled_filter(!isnothing, xs)
    known(Traits.use_generic_broadcasting(xs)) && return GenericBroadcastOp()

    dev = get_device_type(xs)
    dev <: AbstractGPUDevice && return GPUBroadcastOp{dev}()

    # This check needs to be done after the GPU Check
    known(Utils.unrolled_any(!Traits.fast_scalar_indexing, xs)) &&
        return GenericBroadcastOp()
    return LoopedArrayOp()
end
internal_operation_mode(x::AbstractArray) = internal_operation_mode((x,))

CRC.@non_differentiable internal_operation_mode(::Any...)
