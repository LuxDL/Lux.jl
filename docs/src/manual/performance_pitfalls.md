# Performance Pitfalls & How to Catch Them

Go through the following documentations for general performance tips:

1. [Official Julia Performance Tips](https://docs.julialang.org/en/v1/manual/performance-tips/).
2. [Recommendations for selecting AD packages](@ref autodiff-recommendations).

## Spurious Type-Promotion

Lux by-default uses Julia semantics for type-promotions, while this means that we do the
"correct" numerical thing, this can often come as a surprise to users coming from a more
deep learning background. For example, consider the following code:

```@example spurious-type-promotion
using Lux, Random

rng = Xoshiro(0)

model = Dense(2 => 2, gelu)
ps, st = Lux.setup(rng, model)
Lux.recursive_eltype((ps, st))
```

As we can see that `ps` and `st` are structures with the highest precision being `Float32`.
Now let's run the model using some random data:

```@example spurious-type-promotion
x = rand(rng, 2, 4)

eltype(first(model(x, ps, st)))
```

Oops our output became `Float64`. This will be bad on CPUs but an absolute performance
disaster on GPUs. The reason this happened is that our input `x` was `Float64`. Instead,
we should have used `Float32` input:

```@example spurious-type-promotion
x = rand(rng, Float32, 2, 4)

eltype(first(model(x, ps, st)))
```

This was easy to fix for a small model. But certain layers might incorrectly promote
objects to a higher precision. This will cause a regression in performance. There are 2
recommendations to fix this or track them down:

1. Use [`Lux.Experimental.@debug_mode`](@ref debug-lux-layers) to see which layer is causing
   the type-promotion.
2. Alternatively to control the global behavior of eltypes in Lux and allow it to
   auto-correct the precision use [`match_eltype`](@ref) and the
   [`eltype_mismatch_handling`](@ref automatic-eltypes-preference) preference.

## Scalar Indexing on GPU Arrays

When running code on GPUs, it is recommended to
[disallow scalar indexing](https://cuda.juliagpu.org/stable/usage/workflow/#UsageWorkflowScalar).
Note that this is disabled by default except in REPL. You can disable it even in REPL mode
using:

```@example perf-pitfalls-scalar-indexing
using GPUArraysCore
GPUArraysCore.allowscalar(false)
```

## Type Instabilities

`Lux.jl` is integrated with `DispatchDoctor.jl` to catch type instabilities. You can easily
enable it by setting the `instability_check` preference. This will help you catch type
instabilities in your code. For more information on how to set preferences, check out
[`Lux.set_dispatch_doctor_preferences!`](@ref).
