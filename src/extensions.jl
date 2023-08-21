# Extension functions
"""
    transform(l; preserve_ps_st::Bool=false, force_preserve::Bool=false)

Convert a Flux Model to Lux Model.

:::warning

`transform` always ingores the `active` field of some of the Flux layers. This is almost
never going to be supported.

:::

## Arguments

  - `l`: Flux l or any generic Julia function / object.

## Keyword Arguments

  - `preserve_ps_st`: Set to `true` to preserve the states and parameters of the l. This
    attempts the best possible way to preserve the original model. But it might fail. If you
    need to override possible failures, set `force_preserve` to `true`.

  - `force_preserve`: Some of the transformations with state and parameters preservation
    haven't been implemented yet, in these cases, if `force_transform` is `false` a warning
    will be printed and a core Lux layer will be returned. Else, it will create a
    [`FluxLayer`](@ref).

# Examples

```julia
import Flux
using Lux, Metalhead, Random

m = ResNet(18)
m2 = transform(m.layers)

x = randn(Float32, 224, 224, 3, 1);

ps, st = Lux.setup(Random.default_rng(), m2);

m2(x, ps, st)
```
"""
function transform end

_maybe_flip_conv_weight(x) = copy(x)

"""
    FluxLayer(layer)

Serves as a compatibility layer between Flux and Lux. This uses `Optimisers.destructure`
API internally.

:::warning

Lux was written to overcome the limitations of `destructure` + `Flux`. It is recommended
to rewrite your l in Lux instead of using this layer.

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
struct FluxLayer{L, RE, I} <: AbstractExplicitLayer
    layer::L
    re::RE
    init_parameters::I
end
