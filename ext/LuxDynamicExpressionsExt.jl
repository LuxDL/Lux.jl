module LuxDynamicExpressionsExt

using ChainRulesCore: ChainRulesCore, NoTangent
using DynamicExpressions: DynamicExpressions, Node, OperatorEnum, eval_grad_tree_array,
                          eval_tree_array
using FastClosures: @closure
using ForwardDiff: ForwardDiff
using Lux: Lux, NAME_TYPE, Chain, Parallel, WrappedFunction, DynamicExpressionsLayer
using LuxDeviceUtils: LuxCPUDevice

@static if pkgversion(DynamicExpressions) ≥ v"0.19"
    using DynamicExpressions: EvalOptions

    const EvalOptionsTypes = Union{Missing, EvalOptions, NamedTuple}
else
    const EvalOptionsTypes = Union{Missing, NamedTuple}
end

const CRC = ChainRulesCore

@inline Lux._is_extension_loaded(::Val{:DynamicExpressions}) = true

function Lux.DynamicExpressionsLayer(operator_enum::OperatorEnum, expressions::Node...;
        name::NAME_TYPE=nothing, eval_options::EvalOptionsTypes=missing,
        turbo::Union{Bool, Val, Missing}=missing,
        bumper::Union{Bool, Val, Missing}=missing)
    eval_options = __construct_eval_options(
        eval_options, __construct_eval_options(turbo, bumper))

    length(expressions) == 1 && return Lux.DynamicExpressionsLayer(
        operator_enum, first(expressions), name, eval_options)
    _name_fn = name === nothing ? Returns(nothing) : @closure(i->"$(name)_$(i)")
    #! format: off
    return Chain(
        Parallel(nothing,
            ntuple(i -> DynamicExpressionsLayer(operator_enum, expressions[i],
                _name_fn(i), eval_options), length(expressions))...),
        WrappedFunction{:direct_call}(Lux.__stack1);
        name="DynamicExpressionsLayer")
    #! format: on
end

function Lux.DynamicExpressionsLayer(
        operator_enum::OperatorEnum, expressions::AbstractVector{<:Node}; kwargs...)
    return Lux.DynamicExpressionsLayer(operator_enum, expressions...; kwargs...)
end

__construct_eval_options(::Missing, ::Missing) = (; turbo=Val(false), bumper=Val(false))
function __construct_eval_options(turbo::Union{Bool, Val}, ::Missing)
    return __construct_eval_options(turbo, Val(false))
end
function __construct_eval_options(::Missing, bumper::Union{Bool, Val})
    return __construct_eval_options(Val(false), bumper)
end
function __construct_eval_options(turbo::Union{Bool, Val}, bumper::Union{Bool, Val})
    Base.depwarn("`bumper` and `turbo` are deprecated. Use `eval_options` instead.",
        :DynamicExpressionsLayer)
    return (; turbo, bumper)
end

__construct_eval_options(::Missing, eval_options::EvalOptionsTypes) = eval_options
__construct_eval_options(eval_options::EvalOptionsTypes, ::Missing) = eval_options
function __construct_eval_options(
        eval_options::EvalOptionsTypes, old_eval_options::EvalOptionsTypes)
    throw(ArgumentError("`eval_options`, `turbo` and `bumper` are mutually exclusive. \
                         Don't specify `eval_options` if you are using `turbo` or \
                         `bumper`."))
end

function Lux.__apply_dynamic_expression_internal(
        de::Lux.DynamicExpressionsLayer, expr, operator_enum, x, ps)
    Lux.__update_expression_constants!(expr, ps)
    @static if pkgversion(DynamicExpressions) ≥ v"0.19"
        eval_options = EvalOptions(; de.eval_options.turbo, de.eval_options.bumper)
        return first(eval_tree_array(expr, x, operator_enum; eval_options))
    else
        return first(eval_tree_array(
            expr, x, operator_enum; de.eval_options.turbo, de.eval_options.bumper))
    end
end

function Lux.__apply_dynamic_expression_rrule(
        de::Lux.DynamicExpressionsLayer, expr, operator_enum, x, ps)
    Lux.__update_expression_constants!(expr, ps)
    _, Jₓ, _ = eval_grad_tree_array(
        expr, x, operator_enum; variable=Val(true), de.eval_options.turbo)
    y, Jₚ, _ = eval_grad_tree_array(
        expr, x, operator_enum; variable=Val(false), de.eval_options.turbo)
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
        expr, value_fn.(x), operator_enum; variable=Val(true), de.eval_options.turbo)
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
    y, Jₚ, _ = eval_grad_tree_array(
        expr, x, operator_enum; variable=Val(false), de.eval_options.turbo)
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
        expr, x_value, operator_enum; variable=Val(true), de.eval_options.turbo)
    y, Jₚ, _ = eval_grad_tree_array(
        expr, x_value, operator_enum; variable=Val(false), de.eval_options.turbo)
    partials = ntuple(
        @closure(i->dropdims(sum(partials_fn.(x, i) .* Jₓ; dims=1); dims=1) .+
                    dropdims(sum(partials_fn.(ps, i) .* Jₚ; dims=1); dims=1)),
        N)

    fT = promote_type(eltype(y), T1, T2, eltype(Jₓ), eltype(Jₚ))
    partials_y = ForwardDiff.Partials{N, fT}.(tuple.(partials...))
    return ForwardDiff.Dual{Tag, fT, N}.(y, partials_y)
end

end
