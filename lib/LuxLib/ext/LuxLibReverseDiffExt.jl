module LuxLibReverseDiffExt

if isdefined(Base, :get_extension)
    using ReverseDiff
    import ReverseDiff: SpecialInstruction, TrackedArray, TrackedReal, decrement_deriv!,
                        increment_deriv!, track, value, special_reverse_exec!,
                        special_forward_exec!, @grad_from_chainrules
else
    using ..ReverseDiff
    import ReverseDiff: SpecialInstruction, TrackedArray, TrackedReal, decrement_deriv!,
                        increment_deriv!, track, value, special_reverse_exec!,
                        special_forward_exec!, @grad_from_chainrules
end
using ChainRulesCore, LuxLib, NNlib
import ChainRulesCore as CRC
import LuxLib: groupnorm, _GROUPNORM_IMPL_FLOAT

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

# Patch Conv for ReverseDiff
# NOTE: @grad_from_chainrules was not working for ConvDims!
for func in (:conv, :depthwiseconv, :∇conv_data, :∇conv_filter),
    xType in (:TrackedArray, :AbstractArray),
    wType in (:TrackedArray, :AbstractArray)

    xType == :AbstractArray && wType == :AbstractArray && continue

    @eval begin
        function NNlib.$(func)(x::$(xType), w::$(wType), cdims::ConvDims; kwargs...)
            return track(NNlib.$(func), x, w, cdims; kwargs...)
        end

        function ReverseDiff.track(::typeof(NNlib.$(func)), x::$(xType), w::$(wType),
                                   cdims::ConvDims; kwargs...)
            tape = ReverseDiff.tape(x, w, cdims)
            output_value, back = CRC.rrule(NNlib.$(func), value(x), value(w), cdims;
                                           kwargs...)
            output = track(output_value, tape)
            function closure(cls_args...; cls_kwargs...)
                return CRC.rrule(NNlib.$(func), value(x), value(w), cdims; kwargs...)
            end
            ReverseDiff.record!(tape, SpecialInstruction, NNlib.$(func), (x, w, cdims),
                                output, (back, closure, kwargs))
            return output
        end

        function special_reverse_exec!(instr::SpecialInstruction{typeof(NNlib.$(func)),
                                                                 <:Tuple{$(xType), $(wType),
                                                                         ConvDims}})
            back_output = instr.cache[1](ReverseDiff.deriv(instr.output))
            input_derivs = back_output[2:end]
            ReverseDiff._add_to_deriv!.(instr.input, input_derivs)
            ReverseDiff.unseed!(instr.output)
            return nothing
        end

        function special_forward_exec!(instr::SpecialInstruction{typeof(NNlib.$(func)),
                                                                 <:Tuple{$(xType), $(wType),
                                                                         ConvDims}})
            ReverseDiff.pull_value!.(instr.input)
            out_value = instr.cache[2](ReverseDiff.value.(instr.input)...;
                                       instr.cache[3]...)
            ReverseDiff.value!(instr.output, out_value)
            return nothing
        end
    end
end

end
