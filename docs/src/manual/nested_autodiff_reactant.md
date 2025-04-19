# [Nested AutoDiff with Reactant](@id nested_autodiff_reactant)

Reactant supports higher-order automatic differentiation (AD) out of the box. If you are
using the `Enzyme.autodiff` APIs everything should automatically work. Let's work through
a few simple examples to see how this works.

```@example nested_ad_reactant
using Lux, LinearAlgebra, StableRNGs, Enzyme, Reactant
using ComponentArrays, FiniteDiff
```

## Loss Function containing Jacobian Computation

```@example nested_ad_reactant
function loss_function1(model, x, ps, st, y)
end
```
