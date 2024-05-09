# Automatic Differentiation

Lux is not an AD package, but it composes well with most of the AD packages available in the
Julia ecosystem. This document lists the current level of support for various AD packages in
Lux. Additionally, we provide some convenience functions for working with AD.

```@meta
CurrentModule = Lux
```

## Overview

| AD Package                                                         | CPU   | GPU   | Nested 2nd Order AD | Support Class |
| :----------------------------------------------------------------- | :---- | :---- | :------------------ | :------------ |
| [`ChainRules.jl`](https://github.com/JuliaDiff/ChainRules.jl)[^cr] | Y     | Y     | Y                   | Tier I        |
| [`Zygote.jl`](https://github.com/FluxML/Zygote.jl)                 | Y     | Y     | Y                   | Tier I        |
| [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl)    | Y     | Y     | Y                   | Tier I        |
| [`ReverseDiff.jl`](https://github.com/JuliaDiff/ReverseDiff.jl)    | Y     | N     | N                   | Tier II       |
| [`Tracker.jl`](https://github.com/FluxML/Tracker.jl)               | Y     | Y     | N                   | Tier II       |
| [`Enzyme.jl`](https://github.com/EnzymeAD/Enzyme.jl)               | U[^q] | U[^q] | U[^q]               | Tier III      |
| [`Tapir.jl`](https://github.com/withbayes/Tapir.jl)                | U[^q] | U[^q] | N                   | Tier IV       |
| [`Diffractor.jl`](https://github.com/JuliaDiff/Diffractor.jl)      | U[^q] | U[^q] | U[^q]               | Tier IV       |

[^q]: This feature is supported downstream, but we don't extensively test it to ensure
      that it works with Lux.

[^cr]: Note that `ChainRules.jl` is not really an AD package, but we have first-class
       support for packages that use `rrules`.

### Support Class

  1. **Tier I**: These packages are fully supported and have been tested extensively. Often
     have special rules to enhance performance. Issues for these backends take the highest
     priority.
  2. **Tier II**: These packages are supported and extensively tested but often don't have
     the best performance. Issues against these backends are less critical, but we fix them
     when possible. (Some specific edge cases, especially with AMDGPU, are known to fail
     here)
  3. **Tier III**: These packages are somewhat tested but expect rough edges. Help us
     add tests for these backends to get them to Tier II status.
  4. **Tier IV**: We don't know if these packages currently work with Lux. We'd love to
     add tests for these backends, but currently these are not our priority.

## Index

```@index
Pages = ["autodiff.md"]
```

## JVP & JVP Wrappers

```@docs
jacobian_vector_product
vector_jacobian_product
```

## Batched AD

```@docs
batched_jacobian
```

## Nested 2nd Order AD

Consult the [manual page on Nested AD](@ref nested_autodiff) for information on nested
automatic differentiation.
