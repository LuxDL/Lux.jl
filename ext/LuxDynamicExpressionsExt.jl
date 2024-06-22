module LuxDynamicExpressionsExt

using ChainRulesCore: ChainRulesCore, NoTangent
using DynamicExpressions: DynamicExpressions, Node, OperatorEnum, eval_grad_tree_array
using FastClosures: @closure
using Lux: Lux, NAME_TYPE, Chain, Parallel, WrappedFunction

const CRC = ChainRulesCore

@inline Lux._is_extension_loaded(::Val{:DynamicExpressions}) = true

function Lux.DynamicExpressionsLayer(
        operator_enum::OperatorEnum, expressions::Node...; name::NAME_TYPE=nothing,
        turbo::Union{Bool, Val}=Val(false), bumper::Union{Bool, Val}=Val(false))
    length(expressions) == 1 && return Lux.DynamicExpressionsLayer(
        operator_enum, first(expressions), name, turbo, bumper)
    _name_fn = name === nothing ? Returns(nothing) : @closure(i->"$(name)_$(i)")
    #! format: off
    return Chain(
        Parallel(nothing,
            ntuple(i -> Lux.DynamicExpressionsLayer(operator_enum, expressions[i],
                _name_fn(i), turbo, bumper), length(expressions))...),
        WrappedFunction{:direct_call}(Lux.__stack1);
        name="DynamicExpressionsLayer")
    #! format: on
end

function Lux.DynamicExpressionsLayer(
        operator_enum::OperatorEnum, expressions::AbstractVector{<:Node}; kwargs...)
    return Lux.DynamicExpressionsLayer(operator_enum, expressions...; kwargs...)
end

function Lux.__apply_dynamic_expression_rrule(
        de::Lux.DynamicExpressionsLayer, expr, operator_enum, x, ps)
    Lux.__update_expression_constants!(expr, ps)
    _, Jₓ, _ = eval_grad_tree_array(expr, x, operator_enum; variable=Val(true), de.turbo)
    y, Jₚ, _ = eval_grad_tree_array(expr, x, operator_enum; variable=Val(false), de.turbo)
    __∇apply_dynamic_expression = @closure Δ -> begin
        ∂x = Jₓ .* reshape(Δ, 1, :)
        ∂ps = Jₚ * Δ
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), ∂x, ∂ps, NoTangent()
    end
    return y, __∇apply_dynamic_expression
end

end
