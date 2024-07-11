# Layers here serve as a compatibility layer between different frameworks. The core
# implementation is present in extensions

## DynamicExpressions.jl
## We could constrain the type of `operator_enum` to be `OperatorEnum` but defining
## custom types in extensions tends to be a PITA
"""
    DynamicExpressionsLayer(operator_enum::OperatorEnum, expressions::Node...;
        name::NAME_TYPE=nothing, turbo::Union{Bool, Val}=Val(false),
        bumper::Union{Bool, Val}=Val(false))
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
  - `turbo`: Use LoopVectorization.jl for faster evaluation
  - `bumper`: Use Bumper.jl for faster evaluation

These options are simply forwarded to `DynamicExpressions.jl`'s `eval_tree_array`
and `eval_grad_tree_array` function.

## Example

```jldoctest
julia> using Lux, Random, DynamicExpressions, Zygote

julia> operators = OperatorEnum(; binary_operators=[+, -, *], unary_operators=[cos]);

julia> x1 = Node(; feature=1);

julia> x2 = Node(; feature=2);

julia> expr_1 = x1 * cos(x2 - 3.2)
x1 * cos(x2 - 3.2)

julia> expr_2 = x2 - x1 * x2 + 2.5 - 1.0 * x1
((x2 - (x1 * x2)) + 2.5) - (1.0 * x1)

julia> layer = DynamicExpressionsLayer(operators, expr_1, expr_2)
DynamicExpressionsLayer(
    layer_1 = Parallel(
        layer_1 = DynamicExpressionsLayer(DynamicExpressions.OperatorEnumModule.OperatorEnum{Tuple{typeof(+), typeof(-), typeof(*)}, Tuple{typeof(cos)}}((+, -, *), (cos,)), x1 * cos(x2 - 3.2); turbo=Val{false}(), bumper=Val{false}()),  # 1 parameters
        layer_2 = DynamicExpressionsLayer(DynamicExpressions.OperatorEnumModule.OperatorEnum{Tuple{typeof(+), typeof(-), typeof(*)}, Tuple{typeof(cos)}}((+, -, *), (cos,)), ((x2 - (x1 * x2)) + 2.5) - (1.0 * x1); turbo=Val{false}(), bumper=Val{false}()),  # 2 parameters
    ),
    layer_2 = WrappedFunction{:direct_call}(Lux.__stack1),
)         # Total: 3 parameters,
          #        plus 0 states.

julia> ps, st = Lux.setup(Random.default_rng(), layer)
((layer_1 = (layer_1 = (params = Float32[3.2],), layer_2 = (params = Float32[2.5, 1.0],)), layer_2 = NamedTuple()), (layer_1 = (layer_1 = NamedTuple(), layer_2 = NamedTuple()), layer_2 = NamedTuple()))

julia> x = [1.0f0 2.0f0 3.0f0
            4.0f0 5.0f0 6.0f0]
2×3 Matrix{Float32}:
 1.0  2.0  3.0
 4.0  5.0  6.0

julia> layer(x, ps, st)[1] ≈ Float32[0.6967068 -0.4544041 -2.8266668; 1.5 -4.5 -12.5]
true

julia> ∂x, ∂ps, _ = Zygote.gradient(Base.Fix1(sum, abs2) ∘ first ∘ layer, x, ps, st);

julia> ∂x ≈ Float32[-14.0292 54.206482 180.32669; -0.9995737 10.7700815 55.6814]
true

julia> ∂ps.layer_1.layer_1.params ≈ Float32[-6.451908]
true

julia> ∂ps.layer_1.layer_2.params ≈ Float32[-31.0, 90.0]
true
```
"""
@kwdef @concrete struct DynamicExpressionsLayer <: AbstractExplicitLayer
    operator_enum
    expression
    name = nothing
    turbo = Val(false)
    bumper = Val(false)
end

function Base.show(io::IO, l::DynamicExpressionsLayer)
    print(io,
        "DynamicExpressionsLayer($(l.operator_enum), $(l.expression); turbo=$(l.turbo), bumper=$(l.bumper))")
end

function initialparameters(::AbstractRNG, layer::DynamicExpressionsLayer)
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

@inline function (de::DynamicExpressionsLayer)(x::AbstractVector, ps, st)
    y, st_ = de(reshape(x, :, 1), ps, st)
    return vec(y), st_
end

@inline function (de::DynamicExpressionsLayer)(x::AbstractMatrix, ps, st)
    y = match_eltype(de, ps, st, x)
    return (
        __apply_dynamic_expression(
            de, de.expression, de.operator_enum, y, ps.params, get_device(x)),
        st)
end

function __apply_dynamic_expression(
        de::DynamicExpressionsLayer, expr, operator_enum, x, ps, ::LuxCPUDevice)
    __update_expression_constants!(expr, ps)
    return expr(x, operator_enum; de.turbo, de.bumper)
end

function __apply_dynamic_expression_rrule end

function CRC.rrule(::typeof(__apply_dynamic_expression), de::DynamicExpressionsLayer,
        expr, operator_enum, x, ps, ::LuxCPUDevice)
    if !_is_extension_loaded(Val(:DynamicExpressions))
        error("`DynamicExpressions.jl` is not loaded. Please load it before using \
               computing gradient for `DynamicExpressionLayer`.")
    end
    return __apply_dynamic_expression_rrule(de, expr, operator_enum, x, ps)
end

function __apply_dynamic_expression(de, expr, operator_enum, x, ps, dev)
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

!!! note

    If using `Tracker.jl`, the output will always be a regular `Array`.

!!! danger

    `Tracker.jl` sometimes produces incorrect gradients for `SimpleChains.jl` models. As
    such please test your model with FiniteDifferences or Zygote before using `Tracker.jl`
    for your model.
"""
struct SimpleChainsLayer{ToArray, SL, LL <: Union{Nothing, AbstractExplicitLayer}} <:
       AbstractExplicitLayer
    layer::SL
    lux_layer::LL

    function SimpleChainsLayer{ToArray}(layer, lux_layer=nothing) where {ToArray}
        return new{ToArray, typeof(layer), typeof(lux_layer)}(layer, lux_layer)
    end
    function SimpleChainsLayer(layer, ToArray::Union{Bool, Val}=Val(false))
        return new{__unwrap_val(ToArray), typeof(layer), Nothing}(layer, nothing)
    end
end

function Base.show(
        io::IO, ::MIME"text/plain", s::SimpleChainsLayer{ToArray}) where {ToArray}
    _print_wrapper_model(io, "SimpleChainsLayer{$ToArray}", s.lux_layer)
end

@inline initialstates(::AbstractRNG, ::SimpleChainsLayer) = (;)

@inline function (sc::SimpleChainsLayer{false})(x, ps, st)
    y = match_eltype(sc, ps, st, x)
    return __apply_simple_chain(sc.layer, y, ps.params, get_device(x)), st
end

@inline function (sc::SimpleChainsLayer{true})(x, ps, st)
    y = match_eltype(sc, ps, st, x)
    return convert(Array, __apply_simple_chain(sc.layer, y, ps.params, get_device(x))), st
end

@inline __apply_simple_chain(layer, x, ps, ::LuxCPUDevice) = layer(x, ps)

function __apply_simple_chain(layer, x, ps, dev)
    throw(ArgumentError("`SimpleChains.jl` only supports CPU operations. Current device \
                         detected as $(dev)."))
end

# Workaround for SimpleChains not being able to handle some input types
function CRC.rrule(::typeof(__apply_simple_chain), layer, x, ps, ::LuxCPUDevice)
    res, pb = CRC.rrule(layer, x, ps)
    # Safety measure to prevent errors from weird Array types that SimpleChains doesn't support
    __∇apply_simple_chain = @closure Δ -> begin
        ∂layer, ∂x, ∂ps = pb(convert(Array, Δ))
        return NoTangent(), ∂layer, ∂x, ∂ps, NoTangent()
    end
    return res, __∇apply_simple_chain
end
