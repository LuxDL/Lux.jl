module LuxLibReverseDiffExt

using ChainRulesCore: ChainRulesCore
using LuxLib: LuxLib, Optional
using NNlib: NNlib
using ReverseDiff: ReverseDiff, TrackedArray, TrackedVector, TrackedReal,
                   @grad_from_chainrules

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

@inline LuxLib.__value(x::TrackedReal) = ReverseDiff.value(x)
@inline LuxLib.__value(x::TrackedArray) = ReverseDiff.value(x)
@inline LuxLib.__value(x::AbstractArray{<:TrackedReal}) = ReverseDiff.value.(x)

@inline LuxLib.__aos_to_soa(x::TrackedArray) = x
@inline function LuxLib.__aos_to_soa(x::AbstractArray{<:TrackedReal})
    return reshape(reduce(vcat, x), size(x))
end

# Normalization is type unstable for ReverseDiff so we skip dispatch doctor
for xType in (AbstractArray, TrackedArray),
    scType in (Nothing, AbstractVector, TrackedVector),
    bType in (Nothing, AbstractVector, TrackedVector)

    x_tracked = xType !== TrackedArray
    sc_tracked = scType !== TrackedArray
    b_tracked = bType !== TrackedArray

    !x_tracked && !sc_tracked && !b_tracked && continue

    @eval function LuxLib._normalization(
            x::$xType, running_mean::$scType, running_var::$scType,
            scale::$bType, bias::$bType, reduce_dims::Val,
            training::Val, momentum, epsilon, act::F=identity) where {F}
        return LuxLib.__normalization(x, running_mean, running_var, scale, bias,
            reduce_dims, training, momentum, epsilon, act)
    end
end

end
