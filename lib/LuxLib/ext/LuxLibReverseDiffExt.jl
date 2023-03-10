module LuxLibReverseDiffExt

if isdefined(Base, :get_extension)
    using ReverseDiff
    import ReverseDiff: TrackedArray, TrackedReal, decrement_deriv!, increment_deriv!,
                        value, @grad_from_chainrules
else
    using ..ReverseDiff
    import ReverseDiff: TrackedArray, TrackedReal, decrement_deriv!, increment_deriv!,
                        value, @grad_from_chainrules
end
using ChainRulesCore, LuxLib
import LuxLib: groupnorm, _GROUPNORM_IMPL_FLOAT
import ChainRulesCore as CRC

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

LuxLib._get_device(x::TrackedArray) = LuxLib._get_device(value(x))

# api/dropout.jl
LuxLib._dropout_fptype(x::TrackedArray) = LuxLib._dropout_fptype(value(x))

end
