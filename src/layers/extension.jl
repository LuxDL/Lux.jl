# Layers here serve as a compatibility layer between different frameworks. The core
# implementation is present in extensions
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
@concrete struct FluxLayer <: AbstractLuxLayer
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
    SimpleChainsLayer(layer, to_array::Union{Bool, Val}=Val(false))
    SimpleChainsLayer(layer, lux_layer, to_array)

Wraps a `SimpleChains` layer into a `Lux` layer. All operations are performed using
`SimpleChains` but the layer satisfies the `AbstractLuxLayer` interface.

`ToArray` is a boolean flag that determines whether the output should be converted to a
regular `Array` or not. Default is `false`.

## Arguments

  - `layer`: SimpleChains layer
  - `lux_layer`: Potentially equivalent Lux layer that is used for printing
"""
@concrete struct SimpleChainsLayer <: AbstractLuxLayer
    layer
    lux_layer <: Union{Nothing,AbstractLuxLayer}
    to_array <: StaticBool
end

function SimpleChainsLayer(layer, to_array::BoolType=False())
    return SimpleChainsLayer(layer, nothing, static(to_array))
end

function Base.show(io::IO, ::MIME"text/plain", s::SimpleChainsLayer)
    return PrettyPrinting.print_wrapper_model(io, "SimpleChainsLayer", s.lux_layer)
end

function (sc::SimpleChainsLayer)(x, ps, st)
    y = match_eltype(sc, ps, st, x)
    return (
        to_array(
            sc.to_array,
            apply_simple_chain(sc.layer, y, ps.params, MLDataDevices.get_device(x)),
        ),
        st,
    )
end

to_array(::False, y) = y
to_array(::True, y) = convert(Array, y)

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
        _, ∂x, ∂ps = pb(convert(Array, Utils.recursive_unthunk(Δ)))
        return NoTangent(), NoTangent(), 𝒫x(∂x), 𝒫ps(∂ps), NoTangent()
    end
    return res, ∇apply_simple_chain
end
