module LuxDynamicExpressionsExt

using ChainRulesCore: NoTangent
using DynamicExpressions: eval_grad_tree_array
using FastClosures: @closure
using Lux: Lux

@inline Lux._is_extension_loaded(::Val{:DynamicExpressions}) = true

function Lux.__apply_dynamic_expression_rrule(expr, operator_enum, x, ps)
    Lux.__update_expression_constants!(expr, ps)
    _, Jₓ, _ = eval_grad_tree_array(expr, x, operator_enum; variable=Val(true))
    y, Jₚ, _ = eval_grad_tree_array(expr, x, operator_enum; variable=Val(false))
    __∇apply_dynamic_expression = @closure Δ -> begin
        ∂x = dropdims(
            Lux.batched_mul(reshape(Jₓ, size(Jₓ, 1), 1, :), reshape(Δ, 1, 1, :)); dims=2)
        ∂ps = Jₚ * Δ
        return NoTangent(), NoTangent(), NoTangent(), ∂x, ∂ps, NoTangent()
    end
    return y, __∇apply_dynamic_expression
end

end