# Layers here serve as a compatibility layer between different frameworks. The core
# implementation is present in extensions

## DynamicExpressions.jl
## We could constrain the type of `operator_enum` to be `OperatorEnum` but defining
## custom types in extensions tends to be a PITA
# TODO: Add options for turbo and bumper
@kwdef @concrete struct DynamicExpressionLayer{N <: NAME_TYPE} <: AbstractExplicitLayer
    operator_enum
    expression
    name::N = nothing
end

function initialparameters(::AbstractRNG, layer::DynamicExpressionLayer)
    params = map(Base.Fix2(getproperty, :val),
        filter(node -> node.degree == 0 && node.constant, layer.expression))
    return (; params)
end

function __update_expression_constants!(expression, ps)
    # Don't use `set_constant_refs!` here, since it requires the types to match. In our
    # case we just warn the user
    params = filter(node -> node.degree == 0 && node.constant, expression)
    foreach(enumerate(params)) do (i, node)
        !(node.val isa typeof(ps[i])) &&
            @warn lazy"node.val::$(typeof(node.val)) != ps[$i]::$(typeof(ps[i])). Type of node.val takes precedence. Fix the input expression if this is unintended." maxlog=1
        return node.val = ps[i]
    end
    return
end

@inline function (de::DynamicExpressionLayer)(x, ps, st)
    return (
        __apply_dynamic_expression(
            de.expression, de.operator_enum, x, ps.params, get_device(x)),
        st)
end

function __apply_dynamic_expression(expr, operator_enum, x, ps, ::LuxCPUDevice)
    __update_expression_constants!(expr, ps)
    return expr(x, operator_enum)
end

function __apply_dynamic_expression_rrule end

function CRC.rrule(
        ::typeof(__apply_dynamic_expression), expr, operator_enum, x, ps, ::LuxCPUDevice)
    if !_is_extension_loaded(Val(:DynamicExpressions))
        error("`DynamicExpressions.jl` is not loaded. Please load it before using \
               computing gradient for `DynamicExpressionLayer`.")
    end
    return __apply_dynamic_expression_rrule(expr, operator_enum, x, ps)
end

function __apply_dynamic_expression(expr, operator_enum, x, ps, dev)
    throw(ArgumentError(lazy"`DynamicExpressions.jl` only supports CPU operations. Current device detected as $(dev)."))
end

function DynamicExpressionsLayer(operator_enum, expressions...; name::NAME_TYPE=nothing)
    _name_fn = name === nothing ? i -> nothing : i -> "$(name)_$(i)"
    return Chain(
        Parallel(nothing,
            map(((i, expr),) -> DynamicExpressionLayer(operator_enum, expr, _name_fn(i)),
                enumerate(expressions))...),
        WrappedFunction(@closure(x -> mapfoldl(Base.Fix2(reshape, (1, :)), vcat, x)))) # FIXME: replace this with `stack` once the rrule is fixed upstream
end

function DynamicExpressionsLayer(operator_enum, expressions::AbstractVector; kwargs...)
    return DynamicExpressionsLayer(operator_enum, expressions...; kwargs...)
end

## Flux.jl
"""
    FluxLayer(layer)

Serves as a compatibility layer between Flux and Lux. This uses `Optimisers.destructure`
API internally.

!!! warning

    Lux was written to overcome the limitations of `destructure` + `Flux`. It is
    recommended to rewrite your l in Lux instead of using this layer.

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
    re
    init_parameters
end

Lux.initialparameters(::AbstractRNG, l::FluxLayer) = (p=l.init_parameters(),)

(l::FluxLayer)(x, ps, st) = l.re(ps.p)(x), st

Base.show(io::IO, l::FluxLayer) = print(io, "FluxLayer($(l.layer))")

## SimpleChains.jl

"""
    SimpleChainsLayer{ToArray}(layer)
    SimpleChainsLayer(layer, ToArray::Bool=false)

Wraps a `SimpleChains` layer into a `Lux` layer. All operations are performed using
`SimpleChains` but the layer satisfies the `AbstractExplicitLayer` interface.

`ToArray` is a boolean flag that determines whether the output should be converted to a
regular `Array` or not. Default is `false`.

## Arguments

  - `layer`: SimpleChains layer

!!! note

    Using the 2nd constructor makes the generation of the model struct type unstable.

!!! note

    If using `Tracker.jl`, the output will always be a regular `Array`.
"""
struct SimpleChainsLayer{ToArray, L} <: AbstractExplicitLayer
    layer::L
end

@inline function SimpleChainsLayer{ToArray}(layer) where {ToArray}
    return SimpleChainsLayer{ToArray, typeof(layer)}(layer)
end
@inline SimpleChainsLayer(layer, ToArray::Bool=false) = SimpleChainsLayer{ToArray}(layer)

@inline initialstates(::AbstractRNG, ::SimpleChainsLayer) = (;)

@inline function (sc::SimpleChainsLayer{false})(x, ps, st)
    return __apply_simple_chain(sc.layer, x, ps.params, get_device(x)), st
end

@inline function (sc::SimpleChainsLayer{true})(x, ps, st)
    return convert(Array, __apply_simple_chain(sc.layer, x, ps.params, get_device(x))), st
end

@inline __apply_simple_chain(layer, x, ps, ::LuxCPUDevice) = layer(x, ps)

function __apply_simple_chain(layer, x, ps, dev)
    throw(ArgumentError(lazy"`SimpleChains.jl` only supports CPU operations. Current device detected as $(dev)."))
end

# Workaround for SimpleChains not being able to handle some input types
function CRC.rrule(::typeof(__apply_simple_chain), layer, x, ps, ::LuxCPUDevice)
    res, pb = CRC.rrule(layer, x, ps)
    __∇apply_simple_chain = @closure Δ -> begin
        # Safety measure to prevent errors from weird Array types that SimpleChains doesn't
        # support
        ∂layer, ∂x, ∂ps = pb(convert(Array, Δ))
        return CRC.NoTangent(), ∂layer, ∂x, ∂ps, CRC.NoTangent()
    end
    return res, __∇apply_simple_chain
end
