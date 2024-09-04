# [Updating to Lux v1](@id updating-to-v1)

Lux v1 is a Major Release, mostly to signify the stability of the API. In this page, we list
out a concrete set of changes that need to be made to your code to update to Lux v1. We also
list out some new exciting features that were added as part of this release.

## `LuxLib.jl`

### Breaking Changes

- Old deprecated API with keyword arguments has been removed. See the new docs in [LuxLib
  API](@ref LuxLib-API) for more details.
- Default for [`layernorm`](@ref) dims has been changed to exclude the batch dimension.

## `LuxCore.jl`

### Breaking Changes

- `AbstractExplicitLayer` has been renamed to `AbstractLuxLayer`.
- `AbstractExplicitContainerLayer` behaviour
  - This has been renamed to `AbstractLuxContainerLayer`.
  - Previously, `AbstractExplicitContainerLayer{(:a,)}` (i.e. singleton containers) would
    produce default initial parameters and states without wrapping them in a
    `NamedTuple{(:a,)}`. This was inconsistent with non-singleton containers, and was a
    source of confusion. With `v` we return `(; a = <parameters>)` and `(; a = <states>)`
    by default. See [`AbstractLuxWrapperLayer`](@ref) for a replacement of this
    functionality.
- `inputsize` has been removed since it was ambiguous and not used anywhere.
- Changes to `outputsize`:
  - Single argument version has been removed. See [LuxCore.jl Pull Request
    43](https://github.com/LuxDL/LuxCore.jl/pull/43#issuecomment-2254232817) for more
    details on the rationale behind this change.
  - Fallback implementation has been moved to `Lux.jl`. (i.e. users using Lux shouldn't
    see a difference, but if `Lux.jl` isn't loaded, this function has error.)
    - Internally this uses a `NilArray` that is able to compute sizes without actually
      running the computation.
- `Functors` and `Setfield` have been made into optional dependencies. Certain `LuxCore`
  functionality that rely on these functions, will throw an error if these packages are not
  loaded.

### New Major Features

- Introduction of [`AbstractLuxWrapperLayer`](@ref). This behaves exactly like the old
  singleton container. For example, the old `AbstractExplicitContainerLayer{(:a,)}` is
  equivalent to `AbstractLuxWrapperLayer{:a}`.

## `WeightInitializers.jl`

This was a major release to signify the stability of the API. There were no breaking
changes. We do support a wider range of RNG types, see
[Supported RNG Types](@ref Supported-RNG-Types-WeightInit) for more details.

## `MLDataDevices.jl`

This is the most aggressive change that was made. We renamed the `LuxDeviceUtils.jl` package
to `MLDataDevices.jl`, to allow for non-Lux packages to use this shared device management
abstraction.

!!! warning "Deprecation of `LuxDeviceUtils.jl`"

    This also marks the deprecation of the `LuxDeviceUtils.jl` package. We won't be making
    any updates to that package, including fixing any bugs. All users should switch to
    `MLDataDevices.jl` instead.

### Breaking Changes

- `Lux(___)Device` objects have been renamed to `(___)Device`. For example, `LuxCUDADevice`
  has been renamed to `CUDADevice`.
- `Lux(___)Adaptor` objects have been removed. The corresponding `Device` objects should be
  used directly instead.

### New Major Features

- [`DeviceIterator`](@ref) provides a generalization of `CUDA.CuIterator` and works for all
  backends and more data types (using `Functors.jl`). `MLUtils.DataLoader |> gdev` now
  returns a `DeviceIterator` instead of being a no-op.

## `Lux.jl`

### Breaking Changes (Removed Functionality)

- Direct reexport of `NNlib` has been removed. We reexport selected functionality from
  `NNlib`. Direactly load `NNlib` if you need to use the other functions.
- Flattening of [`Chain`](@ref) layers has been removed, and the corresponding
  `disable_optimizations` kwarg has been removed.
- Some layers overloaded `Base.keys`, these have been removed. These were mostly
  un-documented and weren't supposed to be used outside of the `Lux.jl` package.
- [`Training.TrainState`](@ref) construction with `rng` has been removed.
- Older versions of Preferences have been removed.
- `disable_stacktrace_truncation!` has been removed. From Julia 1.9 onwards, stacktrace
  truncation is enabled by default.
- Certain Experimental features were present outside the `Lux.Experimental` module. These
  have been removed, use them via `Lux.Experimental` instead. Run Julia with with `depwarn`
  as `error` and Lux `v0.5` to see the deprecations.
- `Lux.Experimental.@layer_map` is not longer needed and has been removed. The name of the
  variable prevents writing generic functions and is no longer pre-pended to the `KeyPath`.
  See the docstring of [`Lux.Experimental.layer_map`](@ref) for more details.
- `allow_fast_activation` kwarg has been removed completely. Pass an annonymous function
  as the activation to prevent internal modivations to the activation function.

### Breaking Changes (Moved Functionality)

- `Lux.Experimental.Training` has been moved to `Lux.Training`. We guarantee SemVar
  on this new module.
- `Lux.cpu` and `Lux.gpu` have been removed. Use [`cpu_device`](@ref) and
  [`gpu_device`](@ref) instead.
- `Experimental.@compact` can be directly used via [`@compact`](@ref) now.
- `Experimental.StatefulLuxLayer` has been moved to [`Lux.StatefulLuxLayer`](@ref).
- `st_fixed_path` kwarg has been removed from [`Lux.StatefulLuxLayer`](@ref), instead use it
  as `StatefulLuxLayer{st_fixed_path}(...)`.
- Strings as inputs to [`Experimental.layer_map`](@ref) and
  [`Experimental.@debug_mode`](@ref) are removed, use `Functors.KeyPath` instead.

### Breaking Changes (Changes in Defaults)