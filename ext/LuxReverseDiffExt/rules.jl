# SimpleChains.jl
@grad_from_chainrules Lux.apply_simple_chain(layer, x::TrackedArray, ps, ::LuxCPUDevice)
@grad_from_chainrules Lux.apply_simple_chain(layer, x, ps::TrackedArray, ::LuxCPUDevice)
@grad_from_chainrules Lux.apply_simple_chain(
    layer, x::TrackedArray, ps::TrackedArray, ::LuxCPUDevice)

# DynamicExpressions.jl
@grad_from_chainrules Lux.apply_dynamic_expression(de::Lux.DynamicExpressionsLayer, expr,
    operator_enum, x::TrackedArray, ps, ::LuxCPUDevice)
@grad_from_chainrules Lux.apply_dynamic_expression(de::Lux.DynamicExpressionsLayer, expr,
    operator_enum, x, ps::TrackedArray, ::LuxCPUDevice)
@grad_from_chainrules Lux.apply_dynamic_expression(
    de::Lux.DynamicExpressionsLayer, expr, operator_enum,
    x::TrackedArray, ps::TrackedArray, ::LuxCPUDevice)
