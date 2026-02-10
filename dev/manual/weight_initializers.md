---
url: /dev/manual/weight_initializers.md
---
# Initializing Weights {#Initializing-Weights}

`WeightInitializers.jl` provides common weight initialization schemes for deep learning models.

```julia
using WeightInitializers, Random

# Fixing rng
rng = Random.MersenneTwister(42)
```

```ansi
Random.MersenneTwister(42)
```

```julia
# Explicit rng call
weights = kaiming_normal(rng, 2, 5)
```

```ansi
2×5 Matrix{Float32}:
  0.76545    0.255203  -0.0424012   0.643172   -0.360745
 -0.0499631  0.183381   0.388315   -0.0340666  -0.54248
```

```julia
# Default rng call
weights = kaiming_normal(2, 5)
```

```ansi
2×5 Matrix{Float32}:
 -0.227513  -0.265372   0.265788  1.29955  -0.192836
  0.687611   0.454679  -0.433656  0.20548   0.292002
```

```julia
# Passing kwargs (if needed) with explicit rng call
weights_cl = kaiming_normal(rng; gain=1.0)
weights = weights_cl(2, 5)
```

```ansi
2×5 Matrix{Float32}:
 -0.094564   0.196581   0.0791126  -0.794864  0.631217
 -0.381774  -0.588045  -0.113952   -0.567746  0.261636
```

```julia
# Passing kwargs (if needed) with default rng call
weights_cl = kaiming_normal(; gain=1.0)
weights = weights_cl(2, 5)
```

```ansi
2×5 Matrix{Float32}:
 -0.160876  -0.187646   0.18794   0.918918  -0.136356
  0.486214   0.321506  -0.306641  0.145296   0.206476
```

To generate weights directly on GPU, pass in a `CUDA.RNG`. For a complete list of supported RNG types, see [Supported RNG Types](/api/Building_Blocks/WeightInitializers#Supported-RNG-Types-WeightInit).

```julia
using LuxCUDA

weights = kaiming_normal(CUDA.default_rng(), 2, 5)
```

You can also generate Complex Numbers:

```julia
weights = kaiming_normal(CUDA.default_rng(), ComplexF32, 2, 5)
```

## Quick examples {#Quick-examples}

The package is meant to be working with deep learning libraries such as (F)Lux. All the methods take as input the chosen `rng` type and the dimension for the array.

```julia
weights = init(rng, dims...)
```

The `rng` is optional, if not specified a default one will be used.

```julia
weights = init(dims...)
```

If there is the need to use keyword arguments the methods can be called with just the `rng`  (optionally) and the keywords to get in return a function behaving like the two examples above.

```julia
weights_init = init(rng; kwargs...)
weights = weights_init(rng, dims...)

# Or

weights_init = init(; kwargs...)
weights = weights_init(dims...)
```
