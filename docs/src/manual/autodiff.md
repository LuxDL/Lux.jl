# [Automatic Differentiation](@id autodiff-lux)

Lux is not an AD package, but it composes well with most of the AD packages available in the
Julia ecosystem. This document lists the current level of support for various AD packages in
Lux. Additionally, we provide some convenience functions for working with AD.

## Overview

| AD Package                                                                                                           | Mode              | CPU    | GPU    | TPU | Nested 2nd Order AD | Support Class |
| :------------------------------------------------------------------------------------------------------------------- | :---------------- | :----- | :----- | :-- | :------------------ | :------------ |
| [`Reactant.jl`](https://github.com/EnzymeAD/Reactant.jl)[^re] + [`Enzyme.jl`](https://github.com/EnzymeAD/Enzyme.jl) | Reverse / Forward | ✔️     | ✔️     | ✔️  | ✔️                  | Tier I        |
| [`ChainRules.jl`](https://github.com/JuliaDiff/ChainRules.jl)[^cr]                                                   | Reverse           | ✔️     | ✔️     | ❌  | ✔️                  | Tier I        |
| [`Enzyme.jl`](https://github.com/EnzymeAD/Enzyme.jl)                                                                 | Reverse / Forward | ✔️     | ❓[^q] | ❌  | ❓[^q]              | Tier I[^e]    |
| [`Zygote.jl`](https://github.com/FluxML/Zygote.jl)                                                                   | Reverse           | ✔️     | ✔️     | ❌  | ✔️                  | Tier I        |
| [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl)                                                      | Forward           | ✔️     | ✔️     | ❌  | ✔️                  | Tier I        |
| [`Mooncake.jl`](https://github.com/compintell/Mooncake.jl)                                                           | Reverse           | ❓[^q] | ❌     | ❌  | ❌                  | Tier III      |
| [`ReverseDiff.jl`](https://github.com/JuliaDiff/ReverseDiff.jl)                                                      | Reverse           | ✔️     | ❌     | ❌  | ❌                  | Tier IV       |
| [`Tracker.jl`](https://github.com/FluxML/Tracker.jl)                                                                 | Reverse           | ✔️     | ✔️     | ❌  | ❌                  | Tier IV       |
| [`Diffractor.jl`](https://github.com/JuliaDiff/Diffractor.jl)                                                        | Forward           | ❓[^q] | ❓[^q] | ❌  | ❓[^q]              | Tier IV       |

[^e]:
    Currently Enzyme outperforms other AD packages in terms of CPU performance. However,
    there are some edge cases where it might not work with Lux when not using Reactant. We are working on
    improving the compatibility. Please report any issues you encounter and try Reactant if something fails.

[^q]:
    This feature is supported downstream, but we don't extensively test it to ensure
    that it works with Lux.

[^cr]:
    Note that `ChainRules.jl` is not really an AD package, but we have first-class
    support for packages that use `rrules`.

[^re]:
    Note that `Reactant.jl` is not really an AD package, but a tool for compiling functions, including the use of EnzymeMLIR for AD via `Enzyme.jl`.
    We have first-class support for the usage of `Reactant.jl` for inference and training when using `Enzyme.jl` for differentiation.

## [Recommendations](@id autodiff-recommendations)

- For CPU Use cases:

  1. Use `Reactant.jl` + `Enzyme.jl` for the best performance as well as mutation-support.
     When available, this is the most reliable and fastest option.
  2. Use `Zygote.jl` for the best performance without `Reactant.jl`. This is the most reliable and fastest
     option for CPU for the time-being. (We are working on faster Enzyme support for CPU)
  3. Use `Enzyme.jl`, if there are mutations in the code and/or `Zygote.jl` fails.
  4. If `Enzyme.jl` fails for some reason, (open an issue and) try
     `ReverseDiff.jl` ([possibly with compiled mode](https://juliadiff.org/ReverseDiff.jl/dev/api/#ReverseDiff.compile)).

- For GPU Use cases:

  1. Use `Reactant.jl` + `Enzyme.jl` for the best performance. This is the most reliable and fastest option
     for both NVIDIA and AMD GPUs.
  2. Use `Zygote.jl` for the best performance on non-NVIDIA GPUs if not using Reactant. This is the most reliable and fastest
     non-`Reactant.jl` option for GPU for the time-being. We are working on supporting `Enzyme.jl` without
     `Reactant.jl` for GPU as well.

- For TPU Use cases:

  1. Use `Reactant.jl`. This is the only supported (and fastest) option.

## Support Class

1. **Tier I**: These packages are fully supported and have been tested extensively. Often
   have special rules to enhance performance. Issues for these backends take the highest
   priority.
2. **Tier II**: These packages are supported and extensively tested but often don't have
   the best performance. Issues against these backends are less critical, but we fix them
   when possible. (Some specific edge cases, especially with AMDGPU, are known to fail
   here)
3. **Tier III**: We don't know if these packages currently work with Lux. We'd love to
   add tests for these backends, but currently these are not our priority.
4. **Tier IV**: At some point we may or may not have supported these frameworks. However,
   these are not maintained at the moment. Whatever code exists for these frameworks
   in Lux, may be removed in the future (with a breaking release). It is not recommended
   to use these frameworks with Lux.

## Footnotes
