module LuxLibReverseDiffExt

using ChainRulesCore: ChainRulesCore
using LuxLib: LuxLib
using NNlib: NNlib
using ReverseDiff: ReverseDiff, TrackedArray, TrackedReal, @grad_from_chainrules

const CRC = ChainRulesCore

# Patches: Needs upstreaming (I don't know how to construct an MWE though)
@inline function ReverseDiff.increment_deriv!(
        t::Union{TrackedArray, TrackedReal}, ::CRC.NoTangent, i)
    return ReverseDiff.increment_deriv!(t, zero(eltype(ReverseDiff.value(t))), i)
end
@inline function ReverseDiff.decrement_deriv!(
        t::Union{TrackedArray, TrackedReal}, ::CRC.NoTangent, i)
    return ReverseDiff.decrement_deriv!(t, zero(eltype(ReverseDiff.value(t))), i)
end

# utils.jl
@grad_from_chainrules LuxLib._copy_autodiff_barrier(x::TrackedArray)
@grad_from_chainrules LuxLib._copy_autodiff_barrier(x::TrackedReal)

LuxLib._get_backend(x::TrackedArray) = LuxLib._get_backend(ReverseDiff.value(x))

# api/dropout.jl
LuxLib._dropout_fptype(x::TrackedArray) = LuxLib._dropout_fptype(ReverseDiff.value(x))

# Patch Conv for ReverseDiff
for func in (:conv, :depthwiseconv, :∇conv_data, :∇conv_filter),
    xType in (:AbstractArray, :TrackedArray),
    wType in (:AbstractArray, :TrackedArray)

    LuxLib.__is_tracked(xType, wType) || continue

    @eval @grad_from_chainrules NNlib.$(func)(
        x::$(xType), w::$(wType), cdims::NNlib.ConvDims; kwargs...)
end

# Currently falls back to mapreduce and has a terrible performance
@grad_from_chainrules Base.sum(::typeof(abs2), x::TrackedArray; kwargs...)

for pool in (:maxpool, :meanpool, :lpnormpool)
    @eval @grad_from_chainrules NNlib.$(pool)(x::TrackedArray, ::NNlib.PoolDims; kwargs...)
end

end
