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
