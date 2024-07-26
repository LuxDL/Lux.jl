# Utilities

## Index

```@index
Pages = ["utilities.md"]
```

## [Training API](@id Training-API)

Helper Functions making it easier to train `Lux.jl` models.

Training is meant to be simple and provide extremely basic functionality. We provide
basic building blocks which can be seamlessly composed to create complex training pipelines.

```@docs
Training.TrainState
Training.compute_gradients
Training.apply_gradients
Training.apply_gradients!
Training.single_train_step
Training.single_train_step!
```

## Loss Functions

Loss Functions Objects take 2 forms of inputs:

  1. `ŷ` and `y` where `ŷ` is the predicted output and `y` is the target output.
  2. `model`, `ps`, `st`, `(x, y)` where `model` is the model, `ps` are the parameters,
     `st` are the states and `(x, y)` are the input and target pair. Then it returns the
     loss, updated states, and an empty named tuple. This makes them compatible with the
     [Training API](@ref Training-API).

!!! warning

    When using ChainRules.jl compatible AD (like Zygote), we only compute the gradients
    wrt the inputs and drop any gradients wrt the targets.

```@docs
GenericLossFunction
BinaryCrossEntropyLoss
BinaryFocalLoss
CrossEntropyLoss
DiceCoeffLoss
FocalLoss
HingeLoss
HuberLoss
KLDivergenceLoss
MAELoss
MSELoss
MSLELoss
PoissonLoss
SiameseContrastiveLoss
SquaredHingeLoss
```

## Weight Initialization

!!! warning

    For API documentation on Initialization check out the
    [WeightInitializers.jl](@ref WeightInitializers-API)

## Miscellaneous Utilities

```@docs
Lux.foldl_init
Lux.istraining
Lux.multigate
Lux.xlogy
Lux.xlogx
```

## Recursive Operations

```@docs
Lux.recursive_map
Lux.recursive_add!!
Lux.recursive_copyto!
Lux.recursive_eltype
Lux.recursive_make_zero
Lux.recursive_make_zero!!
```

## Updating Floating Point Precision

By default, Lux uses Float32 for all parameters and states. To update the precision
simply pass the parameters / states / arrays into one of the following functions.

```@docs
Lux.f16
Lux.f32
Lux.f64
```

## Element Type Matching

```@docs
match_eltype
```

## Stateful Layer

```@docs
StatefulLuxLayer
```

## Compact Layer

```@docs
@compact
@init_fn
@non_trainable
```

## Truncated Stacktraces (Deprecated)

```@docs
Lux.disable_stacktrace_truncation!
```

## Device Management / Data Transfer (Deprecated)

```@docs
Lux.cpu
Lux.gpu
```

!!! warning

    For detailed API documentation on Data Transfer check out the
    [LuxDeviceUtils.jl](@ref LuxDeviceUtils-API)
