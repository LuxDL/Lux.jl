# [Supporting Both Flux and Lux](@id flux-lux-interop)

A common question for package maintainers is: "How can I support both Flux and Lux in my package?"
This guide provides a comprehensive approach to maintaining compatibility with both frameworks
while minimizing code duplication and dependency overhead.

## The Core Strategy

The recommended approach is to:

1. **Define your core layers using LuxCore**: Use `LuxCore.jl` as your primary interface since it's a lighter dependency than full `Lux.jl`
2. **Construct a StatefulLuxLayer**: Wrap the layer in a [`StatefulLuxLayer`](@ref) to provide a Flux-style interface

This strategy allows users to choose their preferred framework while keeping your package's
core functionality framework-agnostic.

## Implementation Pattern

### 1. Core Layer Definition

First, define your layer using the LuxCore interface:

```@example flux_lux_interop
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

### 2. Wrap the layer in a StatefulLuxLayer

[`StatefulLuxLayer`](@ref) is a convenience wrapper over Lux layers which stores the
parameters and states (and handles updating the state internally). This layer is also
compatible with `Flux.jl`.

## Usage Examples

### Using the Lux Interface

```@example flux_lux_interop
using LuxCore, Random, Flux

# Create layer and setup
rng = Random.default_rng()
layer = MyCustomLuxLayer(4, 2; activation=tanh)
ps, st = LuxCore.setup(rng, layer)

# Forward pass
x = randn(Float32, 4, 32)  # batch of 32 samples
y, st_new = layer(x, ps, st)
```

### Using the Flux Interface

```@example flux_lux_interop
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

## Best Practices

1. **Use LuxCore for core definitions**: Depend on `LuxCore.jl` rather than full `Lux.jl` to minimize dependencies.

2. **Lazy loading**: Use package extensions to avoid loading Flux unless needed.

## Common Gotchas

1. **Mutable state in layer structs**: Remember that Lux layers should not contain mutable state. Put mutable objects in the state, not the layer.

2. **Parameter sharing**: Be careful with parameter sharing when converting between interfaces.

3. **Extension loading**: Users need to load Flux explicitly to access the Flux interface, even if your package supports it.

By following this pattern, you can provide excellent support for both Flux and Lux users while maintaining clean, maintainable code.
