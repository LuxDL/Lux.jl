# Layers here serve as a compatibility layer between different frameworks. The core
# implementation is present in extensions

## DynamicExpressions.jl
## We could constrain the type of `operator_enum` to be `OperatorEnum` but defining
## custom types in extensions tends to be a PITA
"""
    DynamicExpressionsLayer(operator_enum::OperatorEnum, expressions::Node...;
        name::NAME_TYPE=nothing, eval_options::EvalOptions=EvalOptions())
    DynamicExpressionsLayer(operator_enum::OperatorEnum,
        expressions::AbstractVector{<:Node}; kwargs...)

Wraps a `DynamicExpressions.jl` `Node` into a Lux layer and allows the constant nodes to
be updated using any of the AD Backends.

For details about these expressions, refer to the
[`DynamicExpressions.jl` documentation](https://symbolicml.org/DynamicExpressions.jl/dev/types/).

## Arguments

  - `operator_enum`: `OperatorEnum` from `DynamicExpressions.jl`
  - `expressions`: `Node` from `DynamicExpressions.jl` or `AbstractVector{<:Node}`

## Keyword Arguments

  - `name`: Name of the layer
  - `turbo`: Use LoopVectorization.jl for faster evaluation **(Deprecated)**
  - `bumper`: Use Bumper.jl for faster evaluation **(Deprecated)**
  - `eval_options`: EvalOptions from `DynamicExpressions.jl`

These options are simply forwarded to `DynamicExpressions.jl`'s `eval_tree_array`
and `eval_grad_tree_array` function.

!!! danger "Deprecation Notice"

    These options are deprecated and will be removed in v1. Please use the version in
    [`Boltz.jl`](https://github.com/LuxDL/Boltz.jl) instead.
"""
struct DynamicExpressionsLayer{OE, E, N, EO} <: AbstractExplicitLayer
    operator_enum::OE
    expression::E
    name::N
    eval_options::EO

    function DynamicExpressionsLayer(operator_enum::OE, expression::E, name::N,
            eval_options::EO) where {OE, E, N, EO}
        Base.depwarn(
            "`DynamicExpressionsLayer` is deprecated and will be removed in v1. Please \
             use the corresponding version in `Boltz.jl` instead.",
            :DynamicExpressionsLayer)
        return new{OE, E, N, EO}(operator_enum, expression, name, eval_options)
    end
end

function Base.show(io::IO, l::DynamicExpressionsLayer)
    print(io,
        "DynamicExpressionsLayer($(l.operator_enum), $(l.expression); eval_options=$(l.eval_options))")
end

function initialparameters(::AbstractRNG, layer::DynamicExpressionsLayer)
    params = map(Base.Fix2(getproperty, :val),
        filter(node -> node.degree == 0 && node.constant, layer.expression))
    return (; params)
end

function update_de_expression_constants!(expression, ps)
    # Don't use `set_constant_refs!` here, since it requires the types to match. In our
    # case we just warn the user
    params = filter(node -> node.degree == 0 && node.constant, expression)
    foreach(enumerate(params)) do (i, node)
        (node.val isa typeof(ps[i])) ||
            @warn lazy"node.val::$(typeof(node.val)) != ps[$i]::$(typeof(ps[i])). Type of node.val takes precedence. Fix the input expression if this is unintended." maxlog=1
        return node.val = ps[i]
    end
    return
end

function (de::DynamicExpressionsLayer)(x::AbstractVector, ps, st)
    y, stₙ = de(reshape(x, :, 1), ps, st)
    return vec(y), stₙ
end

# NOTE: Unfortunately we can't use `get_device_type` since it causes problems with
#       ReverseDiff
function (de::DynamicExpressionsLayer)(x::AbstractMatrix, ps, st)
    y = match_eltype(de, ps, st, x)
    return (
        apply_dynamic_expression(
            de, de.expression, de.operator_enum, y, ps.params, MLDataDevices.get_device(x)),
        st)
end

function apply_dynamic_expression_internal end

function apply_dynamic_expression(
        de::DynamicExpressionsLayer, expr, operator_enum, x, ps, ::CPUDevice)
    if !is_extension_loaded(Val(:DynamicExpressions))
        error("`DynamicExpressions.jl` is not loaded. Please load it before using \
               `DynamicExpressionsLayer`.")
    end
    return apply_dynamic_expression_internal(de, expr, operator_enum, x, ps)
end

function ∇apply_dynamic_expression end

function CRC.rrule(::typeof(apply_dynamic_expression), de::DynamicExpressionsLayer,
        expr, operator_enum, x, ps, ::CPUDevice)
    if !is_extension_loaded(Val(:DynamicExpressions))
        error("`DynamicExpressions.jl` is not loaded. Please load it before using \
               `DynamicExpressionsLayer`.")
    end
    return ∇apply_dynamic_expression(de, expr, operator_enum, x, ps)
end

function apply_dynamic_expression(de, expr, operator_enum, x, ps, dev)
    throw(ArgumentError("`DynamicExpressions.jl` only supports CPU operations. Current \
                         device detected as $(dev). CUDA.jl will be supported after \
                         https://github.com/SymbolicML/DynamicExpressions.jl/pull/65 is \
                         merged upstream."))
end

## Flux.jl
"""
    FluxLayer(layer)

Serves as a compatibility layer between Flux and Lux. This uses `Optimisers.destructure`
API internally.

!!! warning

    Lux was written to overcome the limitations of `destructure` + `Flux`. It is
    recommended to rewrite your layer in Lux instead of using this layer.

!!! warning

    Introducing this Layer in your model will lead to type instabilities, given the way
    `Optimisers.destructure` works.

## Arguments

  - `layer`: Flux layer

## Parameters

  - `p`: Flattened parameters of the `layer`
"""
@concrete struct FluxLayer <: AbstractExplicitLayer
    layer
    re <: Optimisers.Restructure
    init_parameters
end

function FluxLayer(l)
    p, re = Optimisers.destructure(l)
    return FluxLayer(l, re, Returns(copy(p)))
end

initialparameters(::AbstractRNG, l::FluxLayer) = (p=l.init_parameters(),)

function (l::FluxLayer)(x, ps, st)
    y = match_eltype(l, ps, st, x)
    return l.re(ps.p)(y), st
end

Base.show(io::IO, ::MIME"text/plain", l::FluxLayer) = print(io, "FluxLayer($(l.layer))")

## SimpleChains.jl

"""
    SimpleChainsLayer{ToArray}(layer, lux_layer=nothing)
    SimpleChainsLayer(layer, ToArray::Union{Bool, Val}=Val(false))

Wraps a `SimpleChains` layer into a `Lux` layer. All operations are performed using
`SimpleChains` but the layer satisfies the `AbstractExplicitLayer` interface.

`ToArray` is a boolean flag that determines whether the output should be converted to a
regular `Array` or not. Default is `false`.

## Arguments

  - `layer`: SimpleChains layer
  - `lux_layer`: Potentially equivalent Lux layer that is used for printing
"""
struct SimpleChainsLayer{ToArray, SL, LL <: Union{Nothing, AbstractExplicitLayer}} <:
       AbstractExplicitLayer
    to_array::ToArray
    layer::SL
    lux_layer::LL

    function SimpleChainsLayer{ToArray}(layer, lux_layer=nothing) where {ToArray}
        to_array = static(ToArray)
        return new{typeof(to_array), typeof(layer), typeof(lux_layer)}(
            to_array, layer, lux_layer)
    end
    function SimpleChainsLayer(layer, ToArray::BoolType=False())
        to_array = static(ToArray)
        return new{typeof(to_array), typeof(layer), Nothing}(to_array, layer, nothing)
    end
end

function Base.show(
        io::IO, ::MIME"text/plain", s::SimpleChainsLayer{ToArray}) where {ToArray}
    PrettyPrinting.print_wrapper_model(
        io, "SimpleChainsLayer{to_array=$ToArray}", s.lux_layer)
end

function (sc::SimpleChainsLayer)(x, ps, st)
    y = match_eltype(sc, ps, st, x)
    return (
        simple_chain_output(
            sc, apply_simple_chain(sc.layer, y, ps.params, MLDataDevices.get_device(x))),
        st)
end

simple_chain_output(::SimpleChainsLayer{False}, y) = y
simple_chain_output(::SimpleChainsLayer{True}, y) = convert(Array, y)

apply_simple_chain(layer, x, ps, ::CPUDevice) = layer(x, ps)

function apply_simple_chain(layer, x, ps, dev)
    throw(ArgumentError("`SimpleChains.jl` only supports CPU operations. Current device \
                         detected as $(dev)."))
end

# Workaround for SimpleChains not being able to handle some input types
function CRC.rrule(::typeof(apply_simple_chain), layer, x, ps, ::CPUDevice)
    𝒫x, 𝒫ps = CRC.ProjectTo(x), CRC.ProjectTo(ps)
    res, pb = CRC.rrule(layer, x, ps)
    # Safety measure to prevent errors from weird Array types that SimpleChains doesn't support
    ∇apply_simple_chain = @closure Δ -> begin
        _, ∂x, ∂ps = pb(convert(Array, Δ))
        return NoTangent(), NoTangent(), 𝒫x(∂x), 𝒫ps(∂ps), NoTangent()
    end
    return res, ∇apply_simple_chain
end
