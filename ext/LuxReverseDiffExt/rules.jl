# SimpleChains.jl
@grad_from_chainrules Lux.apply_simple_chain(layer, x::TrackedArray, ps, ::CPUDevice)
@grad_from_chainrules Lux.apply_simple_chain(layer, x, ps::TrackedArray, ::CPUDevice)
@grad_from_chainrules Lux.apply_simple_chain(
    layer, x::TrackedArray, ps::TrackedArray, ::CPUDevice)

# DynamicExpressions.jl
@grad_from_chainrules Lux.apply_dynamic_expression(
    de::Lux.DynamicExpressionsLayer, expr, operator_enum, x::TrackedArray, ps, ::CPUDevice)
@grad_from_chainrules Lux.apply_dynamic_expression(
    de::Lux.DynamicExpressionsLayer, expr, operator_enum, x, ps::TrackedArray, ::CPUDevice)
@grad_from_chainrules Lux.apply_dynamic_expression(
    de::Lux.DynamicExpressionsLayer, expr, operator_enum,
    x::TrackedArray, ps::TrackedArray, ::CPUDevice)
