module LuxLibReverseDiffExt

using ChainRulesCore: ChainRulesCore
using LuxLib: LuxLib
using NNlib: NNlib
using ReverseDiff: ReverseDiff, TrackedArray, TrackedVector, TrackedReal,
                   @grad_from_chainrules

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

    LuxLib.__is_tracked(xType, wType) || continue

    @eval @grad_from_chainrules NNlib.$(func)(
        x::$(xType), w::$(wType), cdims::NNlib.ConvDims; kwargs...)
end

# Currently falls back to mapreduce and has a terrible performance
@grad_from_chainrules Base.sum(::typeof(abs2), x::TrackedArray; kwargs...)

for pool in (:maxpool, :meanpool, :lpnormpool)
    @eval @grad_from_chainrules NNlib.$(pool)(x::TrackedArray, ::NNlib.PoolDims; kwargs...)
end

LuxLib.__value(x::TrackedReal) = ReverseDiff.value(x)
LuxLib.__value(x::TrackedArray) = ReverseDiff.value(x)
LuxLib.__value(x::AbstractArray{<:TrackedReal}) = ReverseDiff.value.(x)

LuxLib.__value(::Type{<:TrackedReal{T}}) where {T} = LuxLib.__value(T)

LuxLib.__has_tracked_value(::TrackedArray) = true
LuxLib.__has_tracked_value(::AbstractArray{<:TrackedReal}) = true
LuxLib.__has_tracked_value(::TrackedReal) = true

LuxLib.__aos_to_soa(x::TrackedArray) = x
function LuxLib.__aos_to_soa(x::AbstractArray{<:TrackedReal})
    return reshape(reduce(vcat, x), size(x))
end

end
