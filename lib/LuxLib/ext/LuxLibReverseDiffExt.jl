module LuxLibReverseDiffExt

using ChainRulesCore: ChainRulesCore
using LuxLib: LuxLib, Utils, Traits
using NNlib: NNlib
using ReverseDiff: ReverseDiff, TrackedArray, TrackedVector, TrackedReal,
                   @grad_from_chainrules
using Static: True

const CRC = ChainRulesCore

# Patches: Needs upstreaming (I don't know how to construct an MWE though)
function ReverseDiff.increment_deriv!(
        t::Union{TrackedArray, TrackedReal}, ::CRC.NoTangent, i)
    return ReverseDiff.increment_deriv!(t, zero(eltype(ReverseDiff.value(t))), i)
end
function ReverseDiff.decrement_deriv!(
        t::Union{TrackedArray, TrackedReal}, ::CRC.NoTangent, i)
    return ReverseDiff.decrement_deriv!(t, zero(eltype(ReverseDiff.value(t))), i)
end

# Patch Conv for ReverseDiff
for func in (:conv, :depthwiseconv, :∇conv_data, :∇conv_filter),
    xType in (:AbstractArray, :TrackedArray),
    wType in (:AbstractArray, :TrackedArray)

    Utils.is_tracked(xType, wType) || continue

    @eval @grad_from_chainrules NNlib.$(func)(
        x::$(xType), w::$(wType), cdims::NNlib.ConvDims; kwargs...)
end

# batched_mul
@grad_from_chainrules NNlib.batched_mul(
    x::TrackedArray{<:Any, <:Any, 3}, y::TrackedArray{<:Any, <:Any, 3})
@grad_from_chainrules NNlib.batched_mul(
    x::TrackedArray{<:Any, <:Any, 3}, y::AbstractArray{<:Any, 3})
@grad_from_chainrules NNlib.batched_mul(
    x::AbstractArray{<:Any, 3}, y::TrackedArray{<:Any, <:Any, 3})

@grad_from_chainrules LuxLib.Impl.batched_matmul(
    x::TrackedArray{<:Any, <:Any, 3}, y::TrackedArray{<:Any, <:Any, 3})
@grad_from_chainrules LuxLib.Impl.batched_matmul(
    x::TrackedArray{<:Any, <:Any, 3}, y::AbstractArray{<:Any, 3})
@grad_from_chainrules LuxLib.Impl.batched_matmul(
    x::AbstractArray{<:Any, 3}, y::TrackedArray{<:Any, <:Any, 3})

# Currently falls back to mapreduce and has a terrible performance
@grad_from_chainrules Base.sum(::typeof(abs2), x::TrackedArray; kwargs...)

for pool in (:maxpool, :meanpool, :lpnormpool)
    @eval @grad_from_chainrules NNlib.$(pool)(x::TrackedArray, ::NNlib.PoolDims; kwargs...)
end

# Utils extensions
Utils.remove_tracking(x::TrackedReal) = ReverseDiff.value(x)
Utils.remove_tracking(x::TrackedArray) = ReverseDiff.value(x)
Utils.remove_tracking(x::AbstractArray{<:TrackedReal}) = ReverseDiff.value.(x)
Utils.remove_tracking(::Type{<:TrackedReal{T}}) where {T} = Utils.remove_tracking(T)

# Traits extensions
Traits.is_tracked(::Type{<:TrackedReal}) = True()

end
