module LuxTrackerExt

using ADTypes: AutoTracker
using ArrayInterface: ArrayInterface
using ChainRulesCore: ChainRulesCore
using Lux: Lux, LuxCPUDevice
using Lux.Experimental: TrainingBackendCache, TrainState
using LuxCore: LuxCore, AbstractExplicitLayer
using Tracker: Tracker, TrackedArray, TrackedReal, @grad_from_chainrules

const CRC = ChainRulesCore

# Weight Norm Patch
@inline Lux._norm(x::TrackedArray; dims=Colon()) = sqrt.(sum(abs2.(x); dims))

# multigate chain rules
@inline Lux._gate(x::Tracker.TrackedVector, h::Int, n::Int) = x[Lux._gate(h, n)]
@inline Lux._gate(x::Tracker.TrackedMatrix, h::Int, n::Int) = x[Lux._gate(h, n), :]

function __construct_tracked_params(ps, dps)
    map_fn = (p, dp) -> Tracker.TrackedArray(Tracker.Call(), p, dp)
    return Lux.recursive_map(map_fn, ps, dps)
end

# Lux.Training
function Lux.Experimental.compute_gradients(::AutoTracker, obj_fn::F, data,
        ts::TrainState{<:TrainingBackendCache{:Tracker, FT}}) where {F, FT}
    dparams = FT ? ts.cache.dparameters : Lux.recursive_make_zero!!(ts.cache.dparameters)
    ps_tracked = __construct_tracked_params(ts.parameters, dparams)

    loss, st, stats = obj_fn(ts.model, ps_tracked, ts.states, data)
    Tracker.back!(loss)

    ts_new = TrainState(
        TrainingBackendCache{:Tracker, false}(ts.cache.dparameters, nothing), obj_fn,
        ts.model, ts.parameters, st, ts.optimizer, ts.optimizer_state, ts.step)

    return dparams, loss.data, stats, ts_new
end

function Lux.Experimental.compute_gradients(
        ::AutoTracker, obj_fn::F, data, ts::TrainState) where {F}
    grads = Lux.recursive_make_zero(ts.parameters)
    ts_new = TrainState(
        TrainingBackendCache{:Tracker, true}(grads, nothing), obj_fn, ts.model,
        ts.parameters, ts.states, ts.optimizer, ts.optimizer_state, ts.step)
    return Lux.Experimental.compute_gradients(AutoTracker(), obj_fn, data, ts_new)
end

# AoS to SoA conversion
function LuxCore.apply(m::AbstractExplicitLayer, x::AbstractArray{<:TrackedReal}, ps, st)
    @warn "LuxCore.apply(m::AbstractExplicitLayer, \
           x::AbstractArray{<:Tracker.TrackedReal}, ps, st) input was corrected to \
           LuxCore.apply(m::AbstractExplicitLayer, x::Tracker.TrackedArray}, ps, st).\n\n\
           1. If this was not the desired behavior overload the dispatch on `m`.\n\n\
           2. This might have performance implications. Check which layer was causing this \
              problem using `Lux.Experimental.@debug_mode`." maxlog=1
    return LuxCore.apply(m, ArrayInterface.aos_to_soa(x), ps, st)
end

## Prevent an infinite loop
LuxCore.apply(m::AbstractExplicitLayer, x::TrackedArray, ps, st) = m(x, ps, st)

@inline Lux.__eltype(::TrackedArray{T}) where {T} = T
@inline Lux.__eltype(::TrackedReal{T}) where {T} = T
@inline Lux.__eltype(::AbstractArray{<:TrackedReal{T}}) where {T} = T

@inline Lux.__reverse(x::TrackedArray; dims=:) = ArrayInterface.aos_to_soa(reverse(x; dims))
@inline function Lux.__reverse(x::AbstractArray{<:TrackedReal}; dims=:)
    return ArrayInterface.aos_to_soa(reverse(x; dims))
end

# SimpleChains.jl: DON'T REPLACE THESE WITH @grad_from_chainrules
for T1 in (:TrackedArray, :AbstractArray), T2 in (:TrackedArray, :AbstractArray)
    T1 === :AbstractArray && T2 === :AbstractArray && continue

    @eval function Lux.__apply_simple_chain(layer, x::$(T1), ps::$(T2), dev::LuxCPUDevice)
        return Tracker.track(Lux.__apply_simple_chain, layer, x, ps, dev)
    end
end

Tracker.@grad function Lux.__apply_simple_chain(layer, x, ps, ::LuxCPUDevice)
    @warn "`Tracker.jl` often produces incorrect gradients for `SimpleChains.jl` models. \
           As such please test your model with FiniteDifferences or Zygote before using \
           `Tracker.jl` for your model." maxlog=1
    y, pb_f = CRC.rrule(layer, Tracker.data(x), Tracker.data(ps))
    __∇apply_simple_chain = let pb_f = pb_f
        Δ -> begin
            _, ∂x, ∂ps = pb_f(convert(Array, Tracker.data(Δ)))
            return Tracker.nobacksies(:__apply_simple_chain, (nothing, ∂x, ∂ps, nothing))
        end
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
