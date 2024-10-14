# [Compiling Lux Models using `Reactant.jl`](@id reactant-compilation)

Quoting the Reactant.jl Readme:

> Reactant takes Julia function and compile it into MLIR and run fancy optimizations on top
> of it, including using EnzymeMLIR for automatic differentiation, and create relevant
> executables for CPU/GPU/TPU via XLA. It presently operates as a tracing system. Compiled
> functions will assume the same control flow pattern as was original taken by objects used
> at compile time, and control flow (e.g. if, for) as well as any type instabilities will be
> removed. The benefits of this approach is immediately making all such code available for
> advanced optimization with little developer effort.

```@example compile_lux_model
using Lux, Reactant, Enzyme, Random
```
