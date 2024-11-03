# LuxLib

Backend for [Lux.jl](http://lux.csail.mit.edu/).

## Tutorials

This is a developer-facing project and most users **should not** depend on it directly. As
such, we don't have tutorials for this package. Instead, we recommend you check out the
[Lux tutorials](http://lux.csail.mit.edu/).

## What's the distinction from [NNlib.jl](https://github.com/FluxML/NNlib.jl)?

This is currently a place to hold more specialized kernels and layer implementations for
Lux.jl. Anyone is free to move these to NNlib.jl (this package is MIT licensed), but I
probably don't have the time to do so myself. But incase you do, open an issue here and let
me know I will delete the code from this package.

## Changelog

### Updating from v0.2 to v0.3

`groupnorm` with statistics tracking support has been removed.

### Updating from v0.1 to v0.2

Support for `CUDA` has been moved to a weak dependency. If you want to use `CUDA`, you need
to install and load `LuxCUDA` as `using LuxCUDA` or `import LuxCUDA`.
