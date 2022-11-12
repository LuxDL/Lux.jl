# v0.4

## v0.4.23

  - Introduces `allow_fast_activation` for layers which store activation functions.

## v0.4.22

  - `Chain` supports indexing.
  - Introduces an API for freezing model parameters.

## v0.4.20

  - Introduces `Lux.@layer_map` and `Lux.layer_map` for mapping over layers.
  - Allows `fmap`-ping over layers.

## v0.4.19

  - Generic Container layers (like `Chain`, `Parallel`, etc.) can now used custom naming for
    their internal layers.

## v0.4.17

  - Major breaking change in experimental Recurrent Cell Implementations.

## v0.4.14
  - Deprecate `bias` in favor of `use_bias` for `RNNCell`.
  - Add `use_bias` kwarg to `LSTMCell` and `GRUCell`

## v0.4.12

  - Deprecate `bias` in favor of `use_bias` for `Dense` and `Scale`.
  - Deprecate `elementwise_*` and `applyactivation` functions.

## v0.4.11

  - Introduces `Lux.Training` API for less clunky training loops.
  - WeightNorm cannot be invoked on a parameter with all elements equal to 0.

## v0.4.10

  - Fixes Recurrent Model bug on GPUs

## v0.4.8

  - Deprecations
    - `ActivationFunction` -- See reasoning in https://github.com/avik-pal/Lux.jl/issues/71
    - Default `parameterlength` / `statelength` / `initialparameters` / `initialstates` for
      certain types
    - `trainmode` / `testmode` with `mode` argument
    - `Conv`: `bias` deprecated in favor of `use_bias`

## v0.4.7

  - Manual detailing Lux Interface
  - Fixes bug with ComponentArray + Optimiser
    https://github.com/FluxML/Optimisers.jl/issues/91
  - `Dropout` Layers caches `1 / (1 - p)` for minor improvements for forward pass
  - `dropout` has a custom rrule -- significantly improves performance for smaller arrays
  - **WARN** A bug with ComponentArray + CUDA + Optimiser usage was introduced. Please
    update to v0.4.8 or higher.

## v0.4.6

  - Documentation revamped
  - `gpu` and `cpu` give a deprecation warning if used on an `AbstractExplicitLayer`

## v0.4.5

  - Allow Arbitrary Parameter Types

## v0.4.4

  - Updated to support julia v1.6 (test time dependency issues)

## v0.4.3

  - Extending Scale to allow for multiple dimension inputs (https://github.com/avik-pal/Lux.jl/pull/40)

## v0.4.2

  - `SelectDim` is no longer type unstable -- Internal storage for the Layer has been changed
  - `Dropout` & `VariationalDropout` return `NoOpLayer` if the probability of dropout is `0`
  - Code Formatting -- SciMLStyle (https://github.com/avik-pal/Lux.jl/pull/31)

## v0.4.1

  - Fix math rendering in docs
  - Add Setfield compat for v1.0
