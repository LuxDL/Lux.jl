---
url: /dev/api/Testing_Functionality/LuxTestUtils.md
---
# LuxTestUtils {#LuxTestUtils}

::: warning Warning

This is a testing package. Hence, we don't use features like weak dependencies to reduce load times. It is recommended that you exclusively use this package for testing and not add a dependency to it in your main package Project.toml.

:::

Implements utilities for testing **gradient correctness** and **dynamic dispatch** of Lux.jl models.

## Testing using JET.jl {#Testing-using-JET.jl}

```julia
@jet f(args...) call_broken=false opt_broken=false
```

Run JET tests on the function `f` with the arguments `args...`. If `JET.jl` fails to compile, then the macro will be a no-op.

**Keyword Arguments**

* `call_broken`: Marks the test\_call as broken.

* `opt_broken`: Marks the test\_opt as broken.

All additional arguments will be forwarded to `JET.@test_call` and `JET.@test_opt`.

::: tip Tip

Instead of specifying `target_modules` with every call, you can set global target modules using [`jet_target_modules!`](/api/Testing_Functionality/LuxTestUtils#LuxTestUtils.jet_target_modules!).

```julia
using LuxTestUtils

jet_target_modules!(["Lux", "LuxLib"]) # Expects Lux and LuxLib to be present in the module calling `@jet`
```

:::

**Example**

```julia
julia> @jet sum([1, 2, 3]) target_modules=(Base, Core)
Test Passed

julia> @jet sum(1, 1) target_modules=(Base, Core) opt_broken=true call_broken=true
Test Broken
  Expression: #= REPL[21]:1 =# JET.@test_opt target_modules = (Base, Core) sum(1, 1)
```

source

```julia
jet_target_modules!(list::Vector{String}; force::Bool=false)
```

This sets `target_modules` for all JET tests when using [`@jet`](/api/Testing_Functionality/LuxTestUtils#LuxTestUtils.@jet).

source

## Gradient Correctness {#Gradient-Correctness}

```julia
test_gradients(f, args...; skip_backends=[], broken_backends=[], kwargs...)
```

Test the gradients of `f` with respect to `args` using the specified backends. The ground truth gradients are computed using FiniteDiff.jl (unless specified otherwise) on CPU.

| Backend        | ADType              | CPU | GPU | Notes             |
|:-------------- |:------------------- |:--- |:--- |:----------------- |
| Zygote.jl      | `AutoZygote()`      | ✔   | ✔   |                   |
| ForwardDiff.jl | `AutoForwardDiff()` | ✔   | ✖   | `len ≤ 32`        |
| Enzyme.jl      | `AutoEnzyme()`      | ✔   | ✖   | Only Reverse Mode |

**Arguments**

* `f`: The function to test the gradients of.

* `args`: The arguments to test the gradients of. Only `AbstractArray`s are considered for gradient computation. Gradients wrt all other arguments are assumed to be `NoTangent()`.

**Keyword Arguments**

* `skip_backends`: A list of backends to skip.

* `broken_backends`: A list of backends to treat as broken.

* `soft_fail`: If `true`, then the test will be recorded as a `soft_fail` test. This overrides any `broken` kwargs. Alternatively, a list of backends can be passed to `soft_fail` to allow soft\_fail tests for only those backends.

* `enzyme_set_runtime_activity`: If `true`, then activate runtime activity for Enzyme.

* `enable_enzyme_reverse_mode`: If `true`, then enable reverse mode for Enzyme.

* `kwargs`: Additional keyword arguments to pass to `check_approx`.

* `ground_truth_backend`: The backend to use for computing the ground truth gradients. Defaults to `AutoFiniteDiff()`.

* `ground_truth_eltype`: The eltype to use for computing the ground truth gradients. Defaults to `Float64`.

**Example**

```julia
julia> f(x, y, z) = x .+ sum(abs2, y.t) + sum(y.x.z)

julia> x = (; t=rand(10), x=(z=[2.0],))

julia> test_gradients(f, 1.0, x, nothing)

```

source

```julia
@test_gradients(f, args...; kwargs...)
```

See the documentation of [`test_gradients`](/api/Testing_Functionality/LuxTestUtils#LuxTestUtils.test_gradients) for more details. This macro provides correct line information for the failing tests.

source

## Extensions to `@test` {#Extensions-to-@test}

```julia
@test_softfail expr
```

Evaluate `expr` and record a test result. If `expr` throws an exception, the test result will be recorded as an error. If `expr` returns a value, and it is not a boolean, the test result will be recorded as an error.

If the test result is false then the test will be recorded as a broken test, else it will be recorded as a pass.

source
