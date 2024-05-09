module LuxTrackerExt

using ADTypes: AutoTracker
using ArrayInterface: ArrayInterface
using ChainRulesCore: ChainRulesCore
using FastClosures: @closure
using Functors: fmap
using Lux: Lux, LuxCPUDevice
using LuxCore: LuxCore
using Setfield: @set!
using Tracker: Tracker, TrackedArray, @grad_from_chainrules

const CRC = ChainRulesCore

# Type Piracy: Need to upstream
Tracker.param(nt::NamedTuple) = fmap(Tracker.param, nt)
Tracker.param(t::Tuple) = map(Tracker.param, t)
Tracker.param(l::LuxCore.AbstractExplicitLayer) = l

Tracker.zero_grad!(nt::NamedTuple) = fmap(Tracker.zero_grad!, nt)

Tracker.extract_grad!(nt::NamedTuple) = fmap(Tracker.extract_grad!, nt)
Tracker.extract_grad!(t::Tuple) = map(Tracker.extract_grad!, t)
Tracker.extract_grad!(::LuxCore.AbstractExplicitLayer) = nothing

Tracker.data(nt::NamedTuple) = fmap(Tracker.data, nt)
Tracker.data(t::Tuple) = map(Tracker.data, t)

# Weight Norm Patch
@inline Lux._norm(x::TrackedArray; dims=Colon()) = sqrt.(sum(abs2.(x); dims))

# multigate chain rules
@inline Lux._gate(x::Tracker.TrackedVector, h::Int, n::Int) = x[Lux._gate(h, n)]
@inline Lux._gate(x::Tracker.TrackedMatrix, h::Int, n::Int) = x[Lux._gate(h, n), :]

# Lux.Training
function Lux.Experimental.compute_gradients(::AutoTracker, objective_function::F, data,
        ts::Lux.Experimental.TrainState) where {F}
    ps_tracked = fmap(Tracker.param, ts.parameters)
    loss, st, stats = objective_function(ts.model, ps_tracked, ts.states, data)
    Tracker.back!(loss)
    @set! ts.states = st
    grads = fmap(Tracker.grad, ps_tracked)
    return grads, loss, stats, ts
end

# AoS to SoA conversion
function Lux.apply(
        m::Lux.AbstractExplicitLayer, x::AbstractArray{<:Tracker.TrackedReal}, ps, st)
    @warn "Lux.apply(m::Lux.AbstractExplicitLayer, \
           x::AbstractArray{<:Tracker.TrackedReal}, ps, st) input was corrected to \
           Lux.apply(m::Lux.AbstractExplicitLayer, x::Tracker.TrackedArray}, ps, st).\n\n\
           1. If this was not the desired behavior overload the dispatch on `m`.\n\n\
           2. This might have performance implications. Check which layer was causing this \
              problem using `Lux.Experimental.@debug_mode`." maxlog=1
    return Lux.apply(m, ArrayInterface.aos_to_soa(x), ps, st)
end

# SimpleChains.jl: DON'T REPLACE THESE WITH @grad_from_chainrules
for T1 in (:TrackedArray, :AbstractArray), T2 in (:TrackedArray, :AbstractArray)
    T1 === :AbstractArray && T2 === :AbstractArray && continue

    @eval function Lux.__apply_simple_chain(layer, x::$(T1), ps::$(T2), dev::LuxCPUDevice)
        return Tracker.track(Lux.__apply_simple_chain, layer, x, ps, dev)
    end
end

Tracker.@grad function Lux.__apply_simple_chain(layer, x, ps, ::LuxCPUDevice)
    y, pb_f = CRC.rrule(layer, Tracker.data(x), Tracker.data(ps))
    __∇apply_simple_chain = @closure Δ -> begin
        _, ∂x, ∂ps = pb_f(convert(Array, Δ))
        return Tracker.nobacksies(:__apply_simple_chain, (nothing, ∂x, ∂ps, nothing))
    end
    # Tracker is not great at handling arbitrary types, so we convert to Array
    return Array(y), __∇apply_simple_chain
end

# DynamicExpressions.jl
for T1 in (:TrackedArray, :AbstractArray), T2 in (:TrackedArray, :AbstractArray)
    T1 === :AbstractArray && T2 === :AbstractArray && continue

    @eval @grad_from_chainrules Lux.__apply_dynamic_expression(
        de::Lux.DynamicExpressionsLayer, expr, operator_enum,
        x::$(T1), ps::$(T2), dev::LuxCPUDevice)
end

end
