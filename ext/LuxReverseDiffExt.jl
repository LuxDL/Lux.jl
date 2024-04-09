module LuxReverseDiffExt

using ADTypes: AutoReverseDiff
using ArrayInterface: ArrayInterface
using Functors: fmap
using Lux: Lux, LuxCPUDevice
using ReverseDiff: ReverseDiff, TrackedArray, @grad_from_chainrules
using Setfield: @set!

function Lux.Experimental.compute_gradients(::AutoReverseDiff, objective_function::F, data,
        ts::Lux.Experimental.TrainState) where {F}
    tape = ReverseDiff.InstructionTape()
    grads = fmap(zero, ts.parameters)
    ps_tracked = fmap((p, g) -> ReverseDiff.TrackedArray(p, g, tape), ts.parameters, grads)
    loss, st, stats = objective_function(ts.model, ps_tracked, ts.states, data)
    loss.deriv = true
    ReverseDiff.reverse_pass!(tape)
    @set! ts.states = st
    return grads, loss, stats, ts
end

# AoS to SoA conversion
function Lux.apply(
        m::Lux.AbstractExplicitLayer, x::AbstractArray{<:ReverseDiff.TrackedReal}, ps, st)
    @warn "Lux.apply(m::Lux.AbstractExplicitLayer, \
           x::AbstractArray{<:ReverseDiff.TrackedReal}, ps, st) input was corrected to \
           Lux.apply(m::Lux.AbstractExplicitLayer, x::ReverseDiff.TrackedArray}, ps, \
           st).\n\n\
           1. If this was not the desired behavior overload the dispatch on `m`.\n\n\
           2. This might have performance implications. Check which layer was causing this \
              problem using `Lux.Experimental.@debug_mode`." maxlog=1
    return Lux.apply(m, reshape(ArrayInterface.aos_to_soa(x), size(x)), ps, st)
end

## Prevent an infinite loop
Lux.apply(m::Lux.AbstractExplicitLayer, x::TrackedArray, ps, st) = m(x, ps, st)

# Handle SimpleChains
@grad_from_chainrules Lux.__apply_simple_chain(layer, x::TrackedArray, ps, ::LuxCPUDevice)
@grad_from_chainrules Lux.__apply_simple_chain(layer, x, ps::TrackedArray, ::LuxCPUDevice)
@grad_from_chainrules Lux.__apply_simple_chain(
    layer, x::TrackedArray, ps::TrackedArray, ::LuxCPUDevice)

end
