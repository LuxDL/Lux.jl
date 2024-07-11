# SimpleChains.jl: DON'T REPLACE THESE WITH @grad_from_chainrules
for T1 in (:TrackedArray, :AbstractArray), T2 in (:TrackedArray, :AbstractArray)
    T1 === :AbstractArray && T2 === :AbstractArray && continue

    @eval function Lux.apply_simple_chain(layer, x::$(T1), ps::$(T2), dev::CPUDevice)
        return Tracker.track(Lux.apply_simple_chain, layer, x, ps, dev)
    end
end

# DynamicExpressions.jl
for T1 in (:TrackedArray, :AbstractArray), T2 in (:TrackedArray, :AbstractArray)
    T1 === :AbstractArray && T2 === :AbstractArray && continue

    @eval @grad_from_chainrules Lux.apply_dynamic_expression(
        de::Lux.DynamicExpressionsLayer, expr,
        operator_enum, x::$(T1), ps::$(T2), dev::CPUDevice)
end

# Nested AD
@grad_from_chainrules Lux.AutoDiffInternalImpl.batched_jacobian_internal(
    f, backend::AbstractADType, x::TrackedArray)
