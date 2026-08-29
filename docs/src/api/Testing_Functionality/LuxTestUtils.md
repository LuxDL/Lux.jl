```@meta
CollapsedDocStrings = true
```

# LuxTestUtils

!!! warning

    This is a testing package. Hence, we don't use features like weak dependencies to
    reduce load times. It is recommended that you exclusively use this package for
    testing and not add a dependency to it in your main package Project.toml.

Implements utilities for testing **gradient correctness** and **dynamic dispatch**
of Lux.jl models.

## Testing using JET.jl

```@docs
@jet
LuxTestUtils.jet_target_modules!
```

## Gradient Correctness

```@docs
test_gradients
@test_gradients
```

## Extensions to `@test`

```@docs
@test_softfail
```
