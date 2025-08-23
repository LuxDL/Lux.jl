# [Supporting Both Flux and Lux](@id flux-lux-interop)

A common question for package maintainers is: "How can I support both Flux and Lux in my package?"
This guide provides a comprehensive approach to maintaining compatibility with both frameworks
while minimizing code duplication and dependency overhead.

## The Core Strategy

The recommended approach is to:

1. **Define your core layers using LuxCore**: Use `LuxCore.jl` as your primary interface since it's a lighter dependency than full `Lux.jl`
2. **Create Flux wrappers**: Implement stateful wrappers that provide a Flux-style interface
3. **Use package extensions**: Leverage Julia's package extension system to conditionally load Flux support

This strategy allows users to choose their preferred framework while keeping your package's
core functionality framework-agnostic.

## Implementation Pattern

### 1. Core Layer Definition

First, define your layer using the LuxCore interface:

```julia
using LuxCore

struct MyCustomLuxLayer <: LuxCore.AbstractLuxLayer
    # Layer configuration (no mutable state!)
    feature_dim::Int
    output_dim::Int
    activation_fn::Function
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

function LuxCore.initialstates(::AbstractRNG, ::MyCustomLuxLayer)
    return NamedTuple()  # No states needed for this example
end

function (layer::MyCustomLuxLayer)(x, ps, st)
    y = ps.weight * x .+ ps.bias
    y = layer.activation_fn.(y)
    return y, st
end
```

### 2. Flux Wrapper Pattern

Create a general wrapper that can work with any LuxCore layer:

```julia
# This part is general and doesn't need to be defined for every specific layer
mutable struct FluxLuxLayer{L <: LuxCore.AbstractLuxLayer,PS,ST}
    internal_layer::L
    ps::PS
    st::ST # Leave untyped if state type is not fixed
end

function FluxLuxLayer(layer::LuxCore.AbstractLuxLayer, rng=Random.default_rng())
    ps, st = LuxCore.setup(rng, layer)
    return FluxLuxLayer(layer, ps, st)
end

# Convenient constructor for your specific layer
function MyCustomFluxLayer(args...; kwargs...)
    layer = MyCustomLuxLayer(args...; kwargs...)
    return FluxLuxLayer(layer)
end

# Forward pass with stateful interface
function (wrapper::FluxLuxLayer)(x)
    y, st_new = wrapper.internal_layer(x, wrapper.ps, wrapper.st)
    wrapper.st = st_new  # Update internal state
    return y
end
```

### 3. Package Extension for Flux Integration

Create `MyPackageFluxExt.jl` in your `ext/` directory:

```julia
module MyPackageFluxExt

using MyPackage  # Your package
using Flux

# Make the wrapper work with Flux's training infrastructure
Flux.@layer FluxLuxLayer (:ps,)

# Optional: Custom training behavior if needed
function Flux.trainable(m::FluxLuxLayer)
    # Return only the parameters that should be trained
    return (ps = m.ps,)
end

end  # module
```

### 4. Package Setup

In your `Project.toml`, add Flux as a weak dependency:

```toml
[deps]
LuxCore = "bb33d45b-7691-41d6-9220-0943567d0623"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[weakdeps]
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"

[extensions]
MyPackageFluxExt = "Flux"
```

## Usage Examples

### Using the Lux Interface

```julia
using MyPackage, LuxCore, Random

# Create layer and setup
rng = Random.default_rng()
layer = MyCustomLuxLayer(4, 2; activation=tanh)
ps, st = LuxCore.setup(rng, layer)

# Forward pass
x = randn(Float32, 4, 32)  # batch of 32 samples
y, st_new = layer(x, ps, st)
```

### Using the Flux Interface

```julia
using MyPackage, Flux

# Create Flux-style layer
model = MyCustomFluxLayer(4, 2; activation=tanh)

# Use like any Flux layer
x = randn(Float32, 4, 32)
y = model(x)

# Works with Flux training
using Optimisers
opt = Adam(0.01)
model, opt_state = Optimisers.setup(opt, model)

# Training step
loss_fn(m, x, y_target) = Flux.mse(m(x), y_target)
loss, grads = Flux.withgradient(loss_fn, model, x, y_target)
model, opt_state = Optimisers.update!(opt_state, model, grads[1])
```

## Advanced Considerations

### Handling Complex State

For layers with complex state (like batch normalization), you may need more sophisticated
state handling:

```julia
function (wrapper::FluxLuxLayer)(x)
    y, st_new = wrapper.internal_layer(x, wrapper.ps, wrapper.st)

    # Deep copy state if it contains mutable objects
    wrapper.st = deepcopy(st_new)
    # Or use Functors.fmap for more efficient copying:
    # wrapper.st = Functors.fmap(copy, st_new)

    return y
end
```

### Performance Considerations

1. **Load Time**: Flux has higher load time due to bundled AD libraries. Lux loads faster.

   ```julia
   # Typical load times (may vary by system):
   # using Lux     # ~0.5 seconds
   # using Flux    # ~0.9 seconds
   ```

## Best Practices

1. **Use LuxCore for core definitions**: Depend on `LuxCore.jl` rather than full `Lux.jl` to minimize dependencies.

2. **Lazy loading**: Use package extensions to avoid loading Flux unless needed.

## Common Gotchas

1. **Mutable state in layer structs**: Remember that Lux layers should not contain mutable state. Put mutable objects in the state, not the layer.

2. **Parameter sharing**: Be careful with parameter sharing when converting between interfaces.

3. **Extension loading**: Users need to load Flux explicitly to access the Flux interface, even if your package supports it.

By following this pattern, you can provide excellent support for both Flux and Lux users while maintaining clean, maintainable code.
