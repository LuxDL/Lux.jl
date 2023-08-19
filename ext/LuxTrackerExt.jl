module LuxTrackerExt

using ADTypes, ChainRulesCore, Functors, Lux, Setfield, Tracker

# Type Piracy: Need to upstream
Tracker.param(nt::NamedTuple) = fmap(Tracker.param, nt)
Tracker.param(t::Tuple) = map(Tracker.param, t)

Tracker.zero_grad!(nt::NamedTuple) = fmap(Tracker.zero_grad!, nt)

Tracker.extract_grad!(nt::NamedTuple) = fmap(Tracker.extract_grad!, nt)
Tracker.extract_grad!(t::Tuple) = map(Tracker.extract_grad!, t)

Tracker.data(nt::NamedTuple) = fmap(Tracker.data, nt)
Tracker.data(t::Tuple) = map(Tracker.data, t)

# Weight Norm Patch
@inline Lux._norm(x::TrackedArray; dims=Colon()) = sqrt.(sum(abs2.(x); dims))

# multigate chain rules
@inline Lux._gate(x::TrackedVector, h::Int, n::Int) = x[Lux._gate(h, n)]
@inline Lux._gate(x::TrackedMatrix, h::Int, n::Int) = x[Lux._gate(h, n), :]

# Lux.Training
function Lux.Experimental.compute_gradients(::AutoTracker, objective_function::Function, data,
    ts::Lux.Experimental.TrainState)
    ps_tracked = fmap(param, ts.parameters)
    loss, st, stats = objective_function(ts.model, ps_tracked, ts.states, data)
    back!(loss)
    @set! ts.states = st
    grads = fmap(Tracker.grad, ps_tracked)
    return grads, loss, stats, ts
end

end
