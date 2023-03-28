# LuxTestUtils.jl

[![Join the chat at https://julialang.zulipchat.com #machine-learning](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/machine-learning)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](http://lux.csail.mit.edu/dev/)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](http://lux.csail.mit.edu/stable/)

[![CI](https://github.com/LuxDL/LuxTestUtils.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/LuxDL/LuxTestUtils.jl/actions/workflows/CI.yml)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

Utilities for testing [Lux.jl](http://lux.csail.mit.edu/stable).

## Installation

```julia
] add LuxTestUtils
```

> **Warning**
> This is a testing package. Hence, we don't use features like weak dependencies to reduce
  load times. It is recommended that you exclusively use this package for testing and not
  add a dependency to it in your main package Project.toml.

## Exported Functions

### Testing using [JET.jl](https://github.com/aviatesk/JET.jl)

We export a simple macro `@jet` to allow testing your code using JET

```julia
help> @jet

  @jet f(args...) call_broken=false opt_broken=false


  Run JET tests on the function `f` with the arguments `args`. If JET fails to compile or
  julia version is < 1.7, then the macro will be a no-op.

  Keyword Arguments
  ===================

    •  `call_broken`: Marks the test_call as broken.

    •  `opt_broken`: Marks the test_opt as broken.

  All additional arguments will be forwarded to @JET.test_call and @JET.test_opt.

  │ Note
  │
  │  Instead of specifying target_modules with every call, you can set preferences for
  │  target_modules using Preferences.jl. For example, to set `target_modules` to
  │  (Lux, LuxLib) we can run:
  │
  │  using Preferences
  │  
  │  set_preferences!(Base.UUID("ac9de150-d08f-4546-94fb-7472b5760531"),
  │                   "target_modules" => ["Lux", "LuxLib"])

  Example
  =========

  @jet sum([1, 2, 3]) target_modules=(Base, Core)
  
  @jet sum(1, 1) target_modules=(Base, Core) opt_broken=true
```

### Gradient Correctness

```julia
help?> @test_gradients
  @test_gradients f args... [kwargs...]


  Compare the gradients computed by `Zygote.jl` (Reverse Mode AD) against:

    •  `Tracker.jl` (Reverse Mode AD)

    •  `ReverseDiff.jl` (Reverse Mode AD)

    •  `ForwardDiff.jl` (Forward Mode AD)

    •  `FiniteDifferences.jl` (Finite Differences)

  │ Tip
  │
  │  This function is completely compatible with `Test.jl`

  Arguments
  ===========

    •  `f`: The function to test.

    •  `args`...: Inputs to f wrt which the gradients are computed.

  Keyword Arguments
  ===================

    • `gpu_testing`: Disables ForwardDiff, ReverseDiff and FiniteDifferences tests.
      (Default: `false`)

    • `soft_fail`: If `true`, the test will not fail if any of the gradients are incorrect,
      instead it will show up as broken. (Default: `false`)

    • `skip_(tracker|reverse_diff|forward_diff|finite_differences)`: Skip the corresponding
      gradient computation and check. (Default: `false`)

    • `large_arrays_skip_(forward_diff|finite_differences)`: Skip the corresponding
      gradient computation and check for large arrays. (Forward Mode and Finite Differences
      are not efficient for large arrays.) (Default: `true`)

    • `large_array_length`: The length of the array above which the gradient computation is
      considered large. (Default: `25`)

    • `max_total_array_size`: Treat as large array if the total size of all arrays is
      greater than this value. (Default: `100`)

    • `(tracker|reverse_diff|forward_diff|finite_differences)_broken`: Mark the
      corresponding gradient test as broken. (Default: `false`)

  Keyword Arguments for check_approx
  ====================================

    • `atol`: Absolute tolerance for gradient comparisons. (Default: `0.0`)

    • `rtol`: Relative tolerance for gradient comparisons. (Default:
      `atol > 0 ? 0.0 : √eps(typeof(atol))`)

    • `nans`: Whether or not NaNs are considered equal. (Default: `false`)

  Example
  =========

  using LuxTestUtils, Test
  
  x = randn(10)
  
  @testset "Showcase Gradient Testing" begin
      @test_gradients sum abs2 x
  
      @test_gradients prod x
  end
```

Internally, it uses `check_approx` which extends `Base.isapprox` for more common cases. It
follows the exact same function call as `isapprox`.

## Passing Runtime Variables to Macro

Macros operate on the syntax and hence can't directly take variable inputs. To get around
this (and especially because you are not using this package in your core package), we can do
the following:

Say we want to mark the Float16 tests for the sum function as broken.

```julia
using LuxTestUtils

for T in (Float16, Float32, Float64)
    x = rand(T, 10, 1)
    # Use `@eval` to interpolate the runtime variable `T` into the macro call
    @eval @jet sum($x) call_broken=$(T == Float16)
end
```
