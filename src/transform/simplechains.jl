"""
    ToSimpleChainsAdaptor()

Adaptor for converting a Lux Model to SimpleChains. The returned model is still a Lux model,
and satisfies the `AbstractExplicitLayer` interfacem but all internal calculations are
performed using SimpleChains.

:::warning

There is no way to preserve trained parameters and states when converting to
`SimpleChains.jl`.

:::

:::warning

Any kind of initialization function is not preserved when converting to `SimpleChains.jl`.

:::

## Arguments

  - `input_dims`: Tuple of input dimensions excluding the batch dimension. These must be of
    `static` type as `SimpleChains` expects.

## Example

```julia
import SimpleChains: static
using Adapt, Lux, Random

lux_model = Chain(Conv((5, 5), 1 => 6, relu), MaxPool((2, 2)),
    Conv((5, 5), 6 => 16, relu), MaxPool((2, 2)), FlattenLayer(3),
    Chain(Dense(256 => 128, relu), Dense(128 => 84, relu), Dense(84 => 10)))

adaptor = ToSimpleChainsAdaptor((static(28), static(28), static(1)))

simple_chains_model = adapt(adaptor, lux_model) # or adaptor(lux_model)

ps, st = Lux.setup(Random.default_rng(), simple_chains_model)
x = randn(Float32, 28, 28, 1, 1)

simple_chains_model(x, ps, st)
```
"""
@concrete struct ToSimpleChainsAdaptor <: AbstractFromLuxAdaptor
    input_dims
end

"""
    Adapt.adapt(from::ToSimpleChainsAdaptor, L::AbstractExplicitLayer)

Adapt a Flux model to Lux model. See [`ToSimpleChainsAdaptor`](@ref) for more details.
"""
function Adapt.adapt(to::ToSimpleChainsAdaptor, L::AbstractExplicitLayer)
    if Base.get_extension(@__MODULE__, :LuxSimpleChainsExt) === nothing
        error("`ToSimpleChainsAdaptor` requires `SimpleChains.jl` to be loaded.")
    end
    sc_layer = __fix_input_dims_simplechain(__to_simplechains_adaptor(L), to.input_dims)
    return SimpleChainsLayer(sc_layer)
end

function __to_simplechains_adaptor end
function __fix_input_dims_simplechain end

struct SimpleChainsModelConversionError <: Exception
    msg::String
end

function SimpleChainsModelConversionError(layer::AbstractExplicitLayer)
    return SimpleChainsModelConversionError(lazy"Conversion to SimpleChains not supported for $(typeof(layer))")
end

function Base.showerror(io::IO, e::SimpleChainsModelConversionError)
    print(io, "SimpleChainsModelConversionError(", e.msg, ")")
    if !TruncatedStacktraces.VERBOSE[]
        println(io, TruncatedStacktraces.VERBOSE_MSG)
    end
end

"""
    SimpleChainsLayer(layer)

Wraps a `SimpleChains` layer into a `Lux` layer. All operations are performed using
`SimpleChains` but the layer satisfies the `AbstractExplicitLayer` interface.

## Arguments

  - `layer`: SimpleChains layer
"""
@concrete struct SimpleChainsLayer <: AbstractExplicitLayer
    layer
end

initialstates(rng::AbstractRNG, layer::SimpleChainsLayer) = (;)

function (sc::SimpleChainsLayer)(x, ps, st)
    return __apply_simple_chain(sc.layer, x, ps.params, get_device(x)), st
end

__apply_simple_chain(layer, x, ps, ::LuxCPUDevice) = layer(x, ps)

function __apply_simple_chain(layer, x, ps, dev)
    throw(ArgumentError(lazy"`SimpleChains.jl` only supports CPU operations. Current device detected as $(dev)."))
end

# Workaround for SimpleChains not being able to handle some input types
function CRC.rrule(cfg::CRC.RuleConfig{>:HasReverseMode},
        ::typeof(__apply_simple_chain), layer, x, ps, dev)
    res, pb = CRC.rrule_via_ad(cfg, layer, x, ps)
    function __∇apply_simple_chain(Δ)
        # Safety measure to prevent errors from weird Array types that SimpleChains doesn't
        # support
        ∂layer, ∂x, ∂ps = pb(convert(Array, Δ))
        return CRC.NoTangent(), ∂layer, ∂x, ∂ps, CRC.NoTangent()
    end
    return res, __∇apply_simple_chain
end
