---
url: /dev/manual/flux_lux_interop.md
---
# Supporting Both Flux and Lux {#flux-lux-interop}

A common question for package maintainers is: "How can I support both Flux and Lux in my package?" This guide provides a comprehensive approach to maintaining compatibility with both frameworks while minimizing code duplication and dependency overhead.

## The Core Strategy {#The-Core-Strategy}

The recommended approach is to:

1. **Define your core layers using LuxCore**: Use `LuxCore.jl` as your primary interface since it's a lighter dependency than full `Lux.jl`

2. **Construct a StatefulLuxLayer**: Wrap the layer in a [`StatefulLuxLayer`](/api/Building_Blocks/LuxCore#LuxCore.StatefulLuxLayerImpl.StatefulLuxLayer) to provide a Flux-style interface

This strategy allows users to choose their preferred framework while keeping your package's core functionality framework-agnostic.

## Implementation Pattern {#Implementation-Pattern}

### 1. Core Layer Definition {#1.-Core-Layer-Definition}

First, define your layer using the LuxCore interface:

```julia
using LuxCore, Random

struct MyCustomLuxLayer{F} <: AbstractLuxLayer
    # Layer configuration (no mutable state!)
    feature_dim::Int
    output_dim::Int
    activation_fn::F
end

function MyCustomLuxLayer(feature_dim::Int, output_dim::Int; activation=identity)
    return MyCustomLuxLayer(feature_dim, output_dim, activation)
end

# Define the Lux interface
function LuxCore.initialparameters(rng::AbstractRNG, layer::MyCustomLuxLayer)
    return (
        weight = randn(rng, Float32, layer.output_dim, layer.feature_dim),
        bias = zeros(Float32, layer.output_dim)
    )
end

function (layer::MyCustomLuxLayer)(x, ps, st)
    y = ps.weight * x .+ ps.bias
    y = layer.activation_fn.(y)
    return y, st
end
```

### 2. Wrap the layer in a StatefulLuxLayer {#2.-Wrap-the-layer-in-a-StatefulLuxLayer}

[`StatefulLuxLayer`](/api/Building_Blocks/LuxCore#LuxCore.StatefulLuxLayerImpl.StatefulLuxLayer) is a convenience wrapper over Lux layers which stores the parameters and states (and handles updating the state internally). This layer is also compatible with `Flux.jl`.

## Usage Examples {#Usage-Examples}

### Using the Lux Interface {#Using-the-Lux-Interface}

```julia
using LuxCore, Random, Flux

# Create layer and setup
rng = Random.default_rng()
layer = MyCustomLuxLayer(4, 2; activation=tanh)
ps, st = LuxCore.setup(rng, layer)

# Forward pass
x = randn(Float32, 4, 32)  # batch of 32 samples
y, st_new = layer(x, ps, st)
```

```ansi
(Float32[0.8389038 0.9993076 … 0.53918076 0.99998707; 0.9966145 -0.11863235 … 0.9981629 0.058118183], NamedTuple())
```

### Using the Flux Interface {#Using-the-Flux-Interface}

```julia
using Flux, LuxCore, Random

# Create Flux-style layer
model = MyCustomLuxLayer(4, 2; activation=tanh)
ps, st = LuxCore.setup(Random.default_rng(), model)
flux_model = LuxCore.StatefulLuxLayer(model, ps, st)

# Use like any Flux layer
x = randn(Float32, 4, 32)
y_target = randn(Float32, 2, 32)

y = flux_model(x)

# Works with Flux training
using Optimisers
opt = Adam(0.01)
opt_state = Optimisers.setup(opt, flux_model)

# Training step
loss_fn(m, x, y_target) = Flux.mse(m(x), y_target)
loss, grads = Flux.withgradient(loss_fn, flux_model, x, y_target)
opt_state, flux_model = Optimisers.update(opt_state, flux_model, grads[1])
```

```ansi
((model = (), ps = (weight = [32mLeaf(Adam(eta=0.01, beta=(0.9, 0.999), epsilon=1.0e-8), [39m(Float32[-0.00738183 -0.00600417 -0.0319064 -0.000142127; 0.00584962 2.74381f-5 -0.0171094 -0.00460663], Float32[5.44906f-6 3.60496f-6 0.0001018 2.01998f-9; 3.42176f-6 7.52841f-11 2.92727f-5 2.12208f-6], (0.81, 0.998001))[32m)[39m, bias = [32mLeaf(Adam(eta=0.01, beta=(0.9, 0.999), epsilon=1.0e-8), [39m(Float32[-0.0187933, -0.00834907], Float32[3.53183f-5, 6.9706f-6], (0.81, 0.998001))[32m)[39m), st = (), st_any = ()), StatefulLuxLayer{Val{true}, Main.MyCustomLuxLayer{typeof(tanh)}, @NamedTuple{weight::Matrix{Float32}, bias::Vector{Float32}}, @NamedTuple{}}(Main.MyCustomLuxLayer{typeof(tanh)}(4, 2, tanh), (weight = Float32[0.6957868 -0.40701553 -0.73492825 0.10898119; 1.4767253 0.08893849 -0.58276945 -0.5118015], bias = Float32[0.009999999, 0.009999999]), NamedTuple(), nothing, Val{true}()))
```

## Best Practices {#Best-Practices}

1. **Use LuxCore for core definitions**: Depend on `LuxCore.jl` rather than full `Lux.jl` to minimize dependencies.

2. **Lazy loading**: Use package extensions to avoid loading Flux unless needed.

## Common Gotchas {#Common-Gotchas}

1. **Mutable state in layer structs**: Remember that Lux layers should not contain mutable state. Put mutable objects in the state, not the layer.

2. **Parameter sharing**: Be careful with parameter sharing when converting between interfaces.

3. **Extension loading**: Users need to load Flux explicitly to access the Flux interface, even if your package supports it.

By following this pattern, you can provide excellent support for both Flux and Lux users while maintaining clean, maintainable code.
