module Traits

using ArrayInterface: ArrayInterface, can_setindex
using ChainRulesCore: ChainRulesCore
using ForwardDiff: ForwardDiff
using NNlib: NNlib
using Static: True, False, static
using StaticArraysCore: StaticArray
using UnrolledUtilities: unrolled_map

using ..LuxLib: Numeric
using ..Utils: NotaNumber, only_derivative, unrolled_any

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

unwrap_array(x) = x
function unwrap_array(x::AbstractArray)
    parent(x) === x && return x
    return unwrap_array(parent(x))
end

has_dual(_) = False()
has_dual(::Type{<:ForwardDiff.Dual}) = True()

has_float16(_) = False()
has_float16(::Type{<:Float16}) = True()

is_tracked(_) = False()

has_autodiff_value(x) = is_tracked(x) | has_dual(x)

static_isa(::Type{T}) where {T} = Base.Fix2(static_isa, T)
static_isa(x, ::Type{T}) where {T} = static(isa(x, T))

function use_generic_broadcasting(xs::Tuple)
    # Float16 is a bit iffy and reordering operations are not optimal for numerical
    # stability so we use the generic implementation for now.
    xs_unwrapped = unrolled_map(unwrap_array, xs)
    return unrolled_any(has_autodiff_value, xs_unwrapped) |
           unrolled_any(has_float16, xs_unwrapped) |
           unrolled_any(static_isa(StaticArray), xs_unwrapped)
end

activation_intermediate_not_needed(::typeof(identity), ::Type) = True()

function activation_intermediate_not_needed(::F, ::Type{T}) where {F, T}
    return static(isconcretetype(Core.Compiler._return_type(
        only_derivative, Tuple{T, F, NotaNumber})))
end

function activation_has_rrule(::F, ::Type{T}) where {F, T}
    return static(isconcretetype(Core.Compiler._return_type(
        only_derivative, Tuple{T, F, T})))
end

# Which activations can be fused into a single kernel
for act in (:identity, :(NNlib.relu), :abs, :abs2)
    @eval fuse_cpu_activation(::typeof($act)) = True()
end
fuse_cpu_activation(::F) where {F} = False()

end

module System

using ChainRulesCore: ChainRulesCore
using Hwloc: Hwloc
using Static: static, False, True

using ..Utils: is_extension_loaded, safe_minimum

const CRC = ChainRulesCore

# Technically Octavian works fine on non-server AMD CPUs, but for safety we disable it
# on non Intel CPUs.
const INTEL_HARDWARE = @static if Sys.ARCH === :x86_64 || Sys.ARCH === :i686
    try
        using CpuId: CpuId
        static(lowercase(string(CpuId.cpuvendor())) == "intel")
    catch
        @warn "Could not detect cpu vendor via CpuId.jl, assuming not Intel. Open an \
               issue in `LuxLib.jl` if this is unexpected."
        False()
    end
else
    False()
end

const AMD_RYZEN_HARDWARE = @static if Sys.ARCH === :x86_64 || Sys.ARCH === :i686
    try
        using CpuId: CpuId
        static(occursin("ryzen", lowercase(string(CpuId.cpubrand()))))
    catch
        @warn "Could not detect cpu brand via CpuId.jl, assuming not Ryzen. Open an issue \
               in `LuxLib.jl` if this is unexpected."
        False()
    end
else
    False()
end

function is_x86_64()
    @static if Sys.ARCH === :x86_64
        return True()
    else
        return False()
    end
end

CRC.@non_differentiable is_x86_64()

function explicit_blas_loaded()
    return is_extension_loaded(Val(:MKL)) |
           is_extension_loaded(Val(:AppleAccelerate)) |
           is_extension_loaded(Val(:BLISBLAS))
end

CRC.@non_differentiable explicit_blas_loaded()

use_octavian() = is_x86_64() & (INTEL_HARDWARE | AMD_RYZEN_HARDWARE)

CRC.@non_differentiable use_octavian()

const L1CacheSize::Int = safe_minimum(Hwloc.l1cache_sizes(), 0)
const L2CacheSize::Int = safe_minimum(Hwloc.l2cache_sizes(), 0)
const L3CacheSize::Int = safe_minimum(Hwloc.l3cache_sizes(), 0)

# NOTE: some systems might not have L3 cache, so we check whether it fits in L(N - 1) cache
fits_in_l1cache(xs::AbstractArray...) = sum(sizeof, xs) ≤ L1CacheSize
CRC.@non_differentiable fits_in_l1cache(::Any...)

function fits_in_l2cache(xs::AbstractArray...)
    return fits_in_l1cache(xs...) || sum(sizeof, xs) ≤ L2CacheSize
end
CRC.@non_differentiable fits_in_l2cache(::Any...)

function fits_in_l3cache(xs::AbstractArray...)
    return fits_in_l2cache(xs...) || sum(sizeof, xs) ≤ L3CacheSize
end
CRC.@non_differentiable fits_in_l3cache(::Any...)

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
