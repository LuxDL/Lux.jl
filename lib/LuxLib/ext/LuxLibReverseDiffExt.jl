module LuxLibReverseDiffExt

if isdefined(Base, :get_extension)
    using ReverseDiff
    import ReverseDiff: SpecialInstruction,
        TrackedArray,
        TrackedReal,
        decrement_deriv!,
        increment_deriv!,
        track,
        value,
        special_reverse_exec!,
        special_forward_exec!,
        @grad_from_chainrules
else
    using ..ReverseDiff
    import ..ReverseDiff: SpecialInstruction,
        TrackedArray,
        TrackedReal,
        decrement_deriv!,
        increment_deriv!,
        track,
        value,
        special_reverse_exec!,
        special_forward_exec!,
        @grad_from_chainrules
end
using ChainRulesCore, LuxLib
import ChainRulesCore as CRC
import LuxLib: AA, __is_tracked

# Patches: Needs upstreaming
@inline function increment_deriv!(t::Union{TrackedArray, TrackedReal}, ::NoTangent, i)
    return increment_deriv!(t, zero(eltype(value(t))), i)
end
@inline function decrement_deriv!(t::Union{TrackedArray, TrackedReal}, ::NoTangent, i)
    return decrement_deriv!(t, zero(eltype(value(t))), i)
end

# utils.jl
@grad_from_chainrules LuxLib._copy_autodiff_barrier(x::TrackedArray)
@grad_from_chainrules LuxLib._copy_autodiff_barrier(x::TrackedReal)

LuxLib._get_backend(x::TrackedArray) = LuxLib._get_backend(value(x))

# api/dropout.jl
LuxLib._dropout_fptype(x::TrackedArray) = LuxLib._dropout_fptype(value(x))

# Patch Conv for ReverseDiff
for func in (:conv, :depthwiseconv, :∇conv_data, :∇conv_filter),
    xType in (:AbstractArray, :TrackedArray), wType in (:AbstractArray, :TrackedArray)

    __is_tracked(xType, wType) || continue

    @eval @grad_from_chainrules NNlib.$(func)(x::$(xType), w::$(wType), cdims::ConvDims;
        kwargs...)
end

end
