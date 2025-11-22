


<a id='LuxTestUtils'></a>

# LuxTestUtils


:::warning


This is a testing package. Hence, we don't use features like weak dependencies to reduce load times. It is recommended that you exclusively use this package for testing and not add a dependency to it in your main package Project.toml.


:::


Implements utilities for testing **gradient correctness** and **dynamic dispatch** of Lux.jl models.


<a id='Index'></a>

## Index

- [`LuxTestUtils.@jet`](#LuxTestUtils.@jet)
- [`LuxTestUtils.@test_gradients`](#LuxTestUtils.@test_gradients)


<a id='Testing-using-JET.jl'></a>

## Testing using JET.jl

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='LuxTestUtils.@jet' href='#LuxTestUtils.@jet'>#</a>&nbsp;<b><u>LuxTestUtils.@jet</u></b> &mdash; <i>Macro</i>.



```julia
@jet f(args...) call_broken=false opt_broken=false
```

Run JET tests on the function `f` with the arguments `args...`. If `JET` fails to compile or julia version is < 1.7, then the macro will be a no-op.

**Keyword Arguments**

  * `call_broken`: Marks the test_call as broken.
  * `opt_broken`: Marks the test_opt as broken.

All additional arguments will be forwarded to `@JET.test_call` and `@JET.test_opt`.

::: note

Instead of specifying `target_modules` with every call, you can set preferences for `target_modules` using `Preferences.jl`. For example, to set `target_modules` to `(Lux, LuxLib)` we can run:

```julia
using Preferences

set_preferences!(Base.UUID("ac9de150-d08f-4546-94fb-7472b5760531"),
    "target_modules" => ["Lux", "LuxLib"])
```

:::

**Example**

```julia
using LuxTestUtils

@testset "Showcase JET Testing" begin
    @jet sum([1, 2, 3]) target_modules=(Base, Core)

    @jet sum(1, 1) target_modules=(Base, Core) opt_broken=true
end
```

</div>
<br>

<a id='Gradient-Correctness'></a>

## Gradient Correctness

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='LuxTestUtils.@test_gradients' href='#LuxTestUtils.@test_gradients'>#</a>&nbsp;<b><u>LuxTestUtils.@test_gradients</u></b> &mdash; <i>Macro</i>.



```julia
@test_gradients f args... [kwargs...]
```

Compare the gradients computed by Zygote.jl (Reverse Mode AD) against:

  * Tracker.jl (Reverse Mode AD)
  * ReverseDiff.jl (Reverse Mode AD)
  * ForwardDiff.jl (Forward Mode AD)
  * FiniteDifferences.jl (Finite Differences)

!!! tip
    This function is completely compatible with Test.jl


**Arguments**

  * `f`: The function to test.
  * `args...`: Inputs to `f` wrt which the gradients are computed.

**Keyword Arguments**

  * `gpu_testing`: Disables ForwardDiff, ReverseDiff and FiniteDifferences tests. (Default: `false`)
  * `soft_fail`: If `true`, the test will not fail if any of the gradients are incorrect, instead it will show up as broken. (Default: `false`)
  * `skip_(tracker|reverse_diff|forward_diff|finite_differences)`: Skip the corresponding gradient computation and check. (Default: `false`)
  * `large_arrays_skip_(forward_diff|finite_differences)`: Skip the corresponding gradient computation and check for large arrays. (Forward Mode and Finite Differences are not efficient for large arrays.) (Default: `true`)
  * `large_array_length`: The length of the array above which the gradient computation is considered large. (Default: 25)
  * `max_total_array_size`: Treat as large array if the total size of all arrays is greater than this value. (Default: 100)
  * `(tracker|reverse_diff|forward_diff|finite_differences)_broken`: Mark the corresponding gradient test as broken. (Default: `false`)

**Keyword Arguments for `check_approx`**

  * `atol`: Absolute tolerance for gradient comparisons. (Default: `0.0`)
  * `rtol`: Relative tolerance for gradient comparisons. (Default: `atol > 0 ? 0.0 : √eps(typeof(atol))`)
  * `nans`: Whether or not NaNs are considered equal. (Default: `false`)

**Example**

```julia
using LuxTestUtils

x = randn(10)

@testset "Showcase Gradient Testing" begin
    @test_gradients sum abs2 x

    @test_gradients prod x
end
```

</div>
<br>
