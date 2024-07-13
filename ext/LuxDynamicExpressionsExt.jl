module LuxDynamicExpressionsExt

using ChainRulesCore: ChainRulesCore, NoTangent
using DynamicExpressions: DynamicExpressions, Node, OperatorEnum, eval_grad_tree_array
using FastClosures: @closure
using ForwardDiff: ForwardDiff
using Lux: Lux, NAME_TYPE, Chain, Parallel, WrappedFunction, DynamicExpressionsLayer
using LuxDeviceUtils: LuxCPUDevice

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
            ntuple(i -> DynamicExpressionsLayer(operator_enum, expressions[i],
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

# Forward Diff rules
function Lux.__apply_dynamic_expression(de::DynamicExpressionsLayer, expr, operator_enum,
        x::AbstractMatrix{<:ForwardDiff.Dual{Tag, T, N}},
        ps, ::LuxCPUDevice) where {T, N, Tag}
    value_fn(x) = ForwardDiff.value(Tag, x)
    partials_fn(x, i) = ForwardDiff.partials(Tag, x, i)

    Lux.__update_expression_constants!(expr, ps)
    y, Jₓ, _ = eval_grad_tree_array(
        expr, value_fn.(x), operator_enum; variable=Val(true), de.turbo)
    partials = ntuple(
        @closure(i->dropdims(sum(partials_fn.(x, i) .* Jₓ; dims=1); dims=1)), N)

    fT = promote_type(eltype(y), T, eltype(Jₓ))
    partials_y = ForwardDiff.Partials{N, fT}.(tuple.(partials...))
    return ForwardDiff.Dual{Tag, fT, N}.(y, partials_y)
end

function Lux.__apply_dynamic_expression(de::DynamicExpressionsLayer, expr, operator_enum, x,
        ps::AbstractVector{<:ForwardDiff.Dual{Tag, T, N}},
        ::LuxCPUDevice) where {T, N, Tag}
    value_fn(x) = ForwardDiff.value(Tag, x)
    partials_fn(x, i) = ForwardDiff.partials(Tag, x, i)

    Lux.__update_expression_constants!(expr, value_fn.(ps))
    y, Jₚ, _ = eval_grad_tree_array(expr, x, operator_enum; variable=Val(false), de.turbo)
    partials = ntuple(
        @closure(i->dropdims(sum(partials_fn.(ps, i) .* Jₚ; dims=1); dims=1)), N)

    fT = promote_type(eltype(y), T, eltype(Jₚ))
    partials_y = ForwardDiff.Partials{N, fT}.(tuple.(partials...))
    return ForwardDiff.Dual{Tag, fT, N}.(y, partials_y)
end

function Lux.__apply_dynamic_expression(de::DynamicExpressionsLayer, expr, operator_enum,
        x::AbstractMatrix{<:ForwardDiff.Dual{Tag, T1, N}},
        ps::AbstractVector{<:ForwardDiff.Dual{Tag, T2, N}},
        ::LuxCPUDevice) where {T1, T2, N, Tag}
    value_fn(x) = ForwardDiff.value(Tag, x)
    partials_fn(x, i) = ForwardDiff.partials(Tag, x, i)

    ps_value = value_fn.(ps)
    x_value = value_fn.(x)

    Lux.__update_expression_constants!(expr, ps_value)
    _, Jₓ, _ = eval_grad_tree_array(
        expr, x_value, operator_enum; variable=Val(true), de.turbo)
    y, Jₚ, _ = eval_grad_tree_array(
        expr, x_value, operator_enum; variable=Val(false), de.turbo)
    partials = ntuple(
        @closure(i->dropdims(sum(partials_fn.(x, i) .* Jₓ; dims=1); dims=1) .+
                    dropdims(sum(partials_fn.(ps, i) .* Jₚ; dims=1); dims=1)),
        N)

    fT = promote_type(eltype(y), T1, T2, eltype(Jₓ), eltype(Jₚ))
    partials_y = ForwardDiff.Partials{N, fT}.(tuple.(partials...))
    return ForwardDiff.Dual{Tag, fT, N}.(y, partials_y)
end

end
