# Automatic Differentiation

Lux is not an AD package, but it composes well with most of the AD packages available in the
Julia ecosystem. This document list the current level of support for various AD packages in
Lux. Additionally, we provide some convenience functions for working with AD.

```@meta
CurrentModule = Lux
```

## Overview

| AD Package                                                      | CPU   | GPU   | Nested 2nd Order AD | Support Class |
| :-------------------------------------------------------------- | :---- | :---- | :------------------ | :------------ |
| [`ChainRules.jl`](https://github.com/JuliaDiff/ChainRules.jl)   | ✔️     | ✔️     | ✔️                   | Tier I        |
| [`Zygote.jl`](https://github.com/FluxML/Zygote.jl)              | ✔️     | ✔️     | ✔️                   | Tier I        |
| [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl) | ✔️     | ✔️     | ✔️                   | Tier I        |
| [`ReverseDiff.jl`](https://github.com/JuliaDiff/ReverseDiff.jl) | ✔️     | ⨉     | ⨉                   | Tier II       |
| [`Tracker.jl`](https://github.com/FluxML/Tracker.jl)            | ✔️     | ✔️     | ⨉                   | Tier II       |
| [`Enzyme.jl`](https://github.com/EnzymeAD/Enzyme.jl)            | ❓[^q] | ❓[^q] | ❓[^q]               | Tier III      |
| [`Tapir.jl`](https://github.com/withbayes/Tapir.jl)             | ❓[^q] | ❓[^q] | ⨉                   | Tier IV       |
| [`Diffractor.jl`](https://github.com/JuliaDiff/Diffractor.jl)   | ❓[^q] | ❓[^q] | ❓[^q]               | Tier IV       |
| --                                                              | --    | --    | --                  | --            |

[^q] That this feature is supported downstream but we don't extensively test it to ensure
     that it works with Lux.

### Support Class

  1. **Tier I**: These packages are fully supported and have been tested extensively. Often
     have special rules to enhance performance. Issues for these backends take highest
     priority.
  2. **Tier II**: These packages are supported and extensively tested, but often don't have
     the best performance. Issues against these backends are less critical, but we fix them
     when we can.
  3. **Tier III**: These packages are somewhat tested, but expect rough edges. Help us
     add tests for these backends so that we can get them to Tier II status.
  4. **Tier IV**: We don't know if these packages currently work with Lux. We'd love to
     add tests for these backends, but currently these are not our priority.

Note that `ChainRules.jl` is not really an AD package, but we have first-class support for
packages that use `rrules`.

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

Consult the {manual page on Nested AD}(@ref nested_autodiff) for information on nested
automatic differentiation.
