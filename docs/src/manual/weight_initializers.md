# Initializing Weights

`WeightInitializers.jl` provides common weight initialization schemes for deep learning
models.

```@example weight-init
using WeightInitializers, Random

# Fixing rng
rng = Random.MersenneTwister(42)
```

```@example weight-init
# Explicit rng call
weights = kaiming_normal(rng, 2, 5)
```

```@example weight-init
# Default rng call
weights = kaiming_normal(2, 5)
```

```@example weight-init
# Passing kwargs (if needed) with explicit rng call
weights_cl = kaiming_normal(rng; gain=1.0)
weights = weights_cl(2, 5)
```

```@example weight-init
# Passing kwargs (if needed) with default rng call
weights_cl = kaiming_normal(; gain=1.0)
weights = weights_cl(2, 5)
```

To generate weights directly on GPU, pass in a `CUDA.RNG`. For a complete list of supported
RNG types, see [Supported RNG Types](@ref Supported-RNG-Types-WeightInit).

```julia
using LuxCUDA

weights = kaiming_normal(CUDA.default_rng(), 2, 5)
```

You can also generate Complex Numbers:

```julia
weights = kaiming_normal(CUDA.default_rng(), ComplexF32, 2, 5)
```

## Quick examples

The package is meant to be working with deep learning libraries such as (F)Lux. All the
methods take as input the chosen `rng` type and the dimension for the array.

```julia
weights = init(rng, dims...)
```

The `rng` is optional, if not specified a default one will be used.

```julia
weights = init(dims...)
```

If there is the need to use keyword arguments the methods can be called with just the `rng` 
(optionally) and the keywords to get in return a function behaving like the two examples
above.

```julia
weights_init = init(rng; kwargs...)
weights = weights_init(rng, dims...)

# Or

weights_init = init(; kwargs...)
weights = weights_init(dims...)
```
