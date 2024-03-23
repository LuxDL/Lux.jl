"""
    FromFluxAdaptor(preserve_ps_st::Bool=false, force_preserve::Bool=false)

Convert a Flux Model to Lux Model.

:::warning

This always ingores the `active` field of some of the Flux layers. This is almost never
going to be supported.

:::

## Keyword Arguments

  - `preserve_ps_st`: Set to `true` to preserve the states and parameters of the layer. This
    attempts the best possible way to preserve the original model. But it might fail. If you
    need to override possible failures, set `force_preserve` to `true`.

  - `force_preserve`: Some of the transformations with state and parameters preservation
    haven't been implemented yet, in these cases, if `force_transform` is `false` a warning
    will be printed and a core Lux layer will be returned. Else, it will create a
    [`FluxLayer`](@ref).

## Example

```julia
import Flux
using Adapt, Lux, Metalhead, Random

m = ResNet(18)
m2 = adapt(FromFluxAdaptor(), m.layers) # or FromFluxAdaptor()(m.layers)

x = randn(Float32, 224, 224, 3, 1);

ps, st = Lux.setup(Random.default_rng(), m2);

m2(x, ps, st)
```
"""
@kwdef struct FromFluxAdaptor <: AbstractToLuxAdaptor
    preserve_ps_st::Bool = false
    force_preserve::Bool = false
end

"""
    FluxLayer(layer)

Serves as a compatibility layer between Flux and Lux. This uses `Optimisers.destructure`
API internally.

:::warning

Lux was written to overcome the limitations of `destructure` + `Flux`. It is recommended
to rewrite your layer in Lux instead of using this layer.

:::

:::warning

Introducing this Layer in your model will lead to type instabilities, given the way
`Optimisers.destructure` works.

:::

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

"""
    Adapt.adapt(from::FromFluxAdaptor, L)

Adapt a Flux model `L` to Lux model. See [`FromFluxAdaptor`](@ref) for more details.
"""
function Adapt.adapt(from::FromFluxAdaptor, L)
    if Base.get_extension(@__MODULE__, :LuxFluxExt) === nothing
        error("`FromFluxAdaptor` requires Flux.jl to be loaded.")
    end
    return __from_flux_adaptor(L; from.preserve_ps_st, from.force_preserve)
end

function __from_flux_adaptor end

Base.@deprecate transform(l; preserve_ps_st::Bool=false, force_preserve::Bool=false) adapt(
    FromFluxAdaptor(preserve_ps_st, force_preserve), l)

# Extend for AMDGPU in extensions
@inline _maybe_flip_conv_weight(x) = copy(x)

struct FluxModelConversionError <: Exception
    msg::String
end

function Base.showerror(io::IO, e::FluxModelConversionError)
    print(io, "FluxModelConversionError(", e.msg, ")")
    return
end
