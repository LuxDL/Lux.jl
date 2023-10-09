# WeightInitializers

[![Join the chat at https://julialang.zulipchat.com #machine-learning](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/machine-learning)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://lux.csail.mit.edu/dev/api/)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://lux.csail.mit.edu/stable/api/)

[![Build status](https://badge.buildkite.com/ffa2c8c3629cd58322446cddd3e8dcc4f121c28a574ee3e626.svg?branch=main)](https://buildkite.com/julialang/weightinitializers-dot-jl)
[![CI](https://github.com/LuxDL/WeightInitializers.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/LuxDL/WeightInitializers.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/LuxDL/WeightInitializers.jl/branch/main/graph/badge.svg?token=1ZY0A2NPEM)](https://codecov.io/gh/LuxDL/WeightInitializers.jl)
[![Package Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/WeightInitializers)](https://pkgs.genieframework.com?packages=WeightInitializers)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

This package is a light dependency providing common weight initialization schemes for deep learning models.

## Example

These code snippets are just provided to give a high level overview
of the functionalities of the package.

```julia
using WeightInitializers, Random

# Fixing rng
rng = MersenneTwister(42)

# Explicit rng call
weights = kaiming_normal(rng, 2, 5)
#2×5 Matrix{Float32}:
# -0.351662   0.0171745   1.12442   -0.296372   -1.67094
# -0.281053  -0.18941    -0.724099   0.0987538   0.634549

# Default rng call
weights = kaiming_normal(2, 5)
#2×5 Matrix{Float32}:
# -0.227513  -0.265372   0.265788  1.29955  -0.192836
#  0.687611   0.454679  -0.433656  0.20548   0.292002

# Passing kwargs (if needed) with explicit rng call
weights_cl = kaiming_normal(rng; gain=1.0)
weights = weights_cl(rng, 2, 5)
#2×5 Matrix{Float32}:
# 0.484056   0.231723   0.164379   0.306147   0.18365
# 0.0836414  0.666965  -0.396323  -0.711329  -0.382971

# Passing kwargs (if needed) with default rng call
weights_cl = kaiming_normal(; gain=1.0)
weights = weights_cl(2, 5)
#2×5 Matrix{Float32}:
# -0.160876  -0.187646   0.18794   0.918918  -0.136356
#  0.486214   0.321506  -0.306641  0.145296   0.206476
```

## API

The package is meant to be working with deep learning
libraries such as F/Lux. All the methods take as input the chosen `rng` type and the dimension for the array.

```julia
weights = init(rng, dims...)
```

The `rng` is optional, if not specified a default one will be used.

```julia
weights = init(dims...)
```

If there is the need to use keyword arguments the methods can be called with just the `rng` (optionally)
and the keywords to get in return a function behaving like the two examples above.

```julia
weights_init = init(rng; kwargs...)
weights = weights_init(rng, dims...)
# or
weights_init = init(; kwargs...)
weights = weights_init(dims...)
```
