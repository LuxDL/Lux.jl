# SimpleChains.jl
@grad_from_chainrules Lux.__apply_simple_chain(layer, x::TrackedArray, ps, ::LuxCPUDevice)
@grad_from_chainrules Lux.__apply_simple_chain(layer, x, ps::TrackedArray, ::LuxCPUDevice)
@grad_from_chainrules Lux.__apply_simple_chain(
    layer, x::TrackedArray, ps::TrackedArray, ::LuxCPUDevice)

# DynamicExpressions.jl
@grad_from_chainrules Lux.__apply_dynamic_expression(de::Lux.DynamicExpressionsLayer, expr,
    operator_enum, x::TrackedArray, ps, ::LuxCPUDevice)
@grad_from_chainrules Lux.__apply_dynamic_expression(de::Lux.DynamicExpressionsLayer, expr,
    operator_enum, x, ps::TrackedArray, ::LuxCPUDevice)
@grad_from_chainrules Lux.__apply_dynamic_expression(
    de::Lux.DynamicExpressionsLayer, expr, operator_enum,
    x::TrackedArray, ps::TrackedArray, ::LuxCPUDevice)

# Nested AD Handling
## ForwardDiff
for type in (:Gradient, :Jacobian)
    cfgname = Symbol(type, :Config)
    fname = Symbol(lowercase(string(type)))
    internal_fname = Symbol(:__internal_forwarddiff_, fname)

    @eval begin
        @grad_from_chainrules Lux.$(internal_fname)(
            f, cfg::ForwardDiff.$(cfgname), chk::Val, x::TrackedArray, y)
        @grad_from_chainrules Lux.$(internal_fname)(
            f, cfg::ForwardDiff.$(cfgname), chk::Val, x, y::TrackedArray)
        @grad_from_chainrules Lux.$(internal_fname)(
            f, cfg::ForwardDiff.$(cfgname), chk::Val, x::TrackedArray, y::TrackedArray)
    end
end

## All other cases
@grad_from_chainrules Lux.__internal_ad_gradient_call(grad_fn, f, x::TrackedArray, y)
@grad_from_chainrules Lux.__internal_ad_gradient_call(grad_fn, f, x, y::TrackedArray)
@grad_from_chainrules Lux.__internal_ad_gradient_call(
    grad_fn, f, x::TrackedArray, y::TrackedArray)

@grad_from_chainrules Lux.__internal_ad_pullback_call(pullback_fn, f, x::TrackedArray, y, u)
@grad_from_chainrules Lux.__internal_ad_pullback_call(pullback_fn, f, x, y::TrackedArray, u)
@grad_from_chainrules Lux.__internal_ad_pullback_call(
    pullback_fn, f, x::TrackedArray, y::TrackedArray, u)

@grad_from_chainrules Lux.__internal_ad_jacobian_call(
    jac_fn, grad_fn, f, x::TrackedArray, y)
@grad_from_chainrules Lux.__internal_ad_jacobian_call(
    jac_fn, grad_fn, f, x, y::TrackedArray)
@grad_from_chainrules Lux.__internal_ad_jacobian_call(
    jac_fn, grad_fn, f, x::TrackedArray, y::TrackedArray)
