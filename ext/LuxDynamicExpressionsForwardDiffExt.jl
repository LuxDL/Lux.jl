module LuxDynamicExpressionsForwardDiffExt

using DynamicExpressions: eval_grad_tree_array
using FastClosures: @closure
using ForwardDiff: ForwardDiff
using Lux: Lux, LuxCPUDevice, DynamicExpressionsLayer

function Lux.__apply_dynamic_expression(de::DynamicExpressionsLayer, expr, operator_enum,
        x::AbstractMatrix{<:ForwardDiff.Dual{Tag, T, N}},
        ps, ::LuxCPUDevice) where {T, N, Tag}
    Lux.__update_expression_constants!(expr, ps)
    x_value = ForwardDiff.value.(x)
    y, Jₓ, _ = eval_grad_tree_array(
        expr, x_value, operator_enum; variable=Val(true), de.turbo)
    partials = ntuple(
        @closure(i->dropdims(sum(ForwardDiff.partials.(x, i) .* Jₓ; dims=1); dims=1)), N)
    fT = promote_type(eltype(y), T, eltype(Jₓ))
    return ForwardDiff.Dual{
        Tag, fT, N}.(y, ForwardDiff.Partials{N, fT}.(tuple.(partials...)))
end

function Lux.__apply_dynamic_expression(de::DynamicExpressionsLayer, expr, operator_enum, x,
        ps::AbstractVector{<:ForwardDiff.Dual{Tag, T, N}},
        ::LuxCPUDevice) where {T, N, Tag}
    Lux.__update_expression_constants!(expr, ForwardDiff.value.(ps))
    y, Jₚ, _ = eval_grad_tree_array(expr, x, operator_enum; variable=Val(false), de.turbo)
    partials = ntuple(
        @closure(i->dropdims(sum(ForwardDiff.partials.(ps, i) .* Jₚ; dims=1); dims=1)), N)
    fT = promote_type(eltype(y), T, eltype(Jₚ))
    return ForwardDiff.Dual{
        Tag, fT, N}.(y, ForwardDiff.Partials{N, fT}.(tuple.(partials...)))
end

function Lux.__apply_dynamic_expression(de::DynamicExpressionsLayer, expr, operator_enum,
        x::AbstractMatrix{<:ForwardDiff.Dual{Tag, T1, N}},
        ps::AbstractVector{<:ForwardDiff.Dual{Tag, T2, N}},
        ::LuxCPUDevice) where {T1, T2, N, Tag}
    ps_value = ForwardDiff.value.(ps)
    x_value = ForwardDiff.value.(x)
    Lux.__update_expression_constants!(expr, ps_value)
    _, Jₓ, _ = eval_grad_tree_array(
        expr, x_value, operator_enum; variable=Val(true), de.turbo)
    y, Jₚ, _ = eval_grad_tree_array(
        expr, x_value, operator_enum; variable=Val(false), de.turbo)
    partials = ntuple(
        @closure(i->dropdims(sum(ForwardDiff.partials.(x, i) .* Jₓ; dims=1); dims=1) .+
                    dropdims(sum(ForwardDiff.partials.(ps, i) .* Jₚ; dims=1); dims=1)),
        N)
    fT = promote_type(eltype(y), T1, T2, eltype(Jₓ), eltype(Jₚ))
    return ForwardDiff.Dual{
        Tag, fT, N}.(y, ForwardDiff.Partials{N, fT}.(tuple.(partials...)))
end

end
