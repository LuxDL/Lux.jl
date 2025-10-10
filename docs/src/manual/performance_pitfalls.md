# Performance Pitfalls & How to Catch Them

Go through the following documentations for general performance tips:

1. [Official Julia Performance Tips](https://docs.julialang.org/en/v1/manual/performance-tips/).
2. [Recommendations for selecting AD packages](@ref autodiff-recommendations).

!!! tip "Using Reactant?"

    If you are using Lux with Reactant, general concerns with Julia performance like
    type-instabilities don't apply. Reactant automatically optimizes away the type
    instabilities.

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

## Data Loading and Device Transfer

A common pattern for loading data and transferring data to GPUs looks like this:

```julia
dataloader = DataLoader(dataset; parallel=true, batchsize=12)  # from MLUtils.jl
gdev = gpu_device()

for (X, y) in dataloader
    X = X |> gdev
    y = y |> gdev
    # ...
    # do some computation
    # ...
end
```

This is typically fast enough, but the data transfer to the device is happening in main
process, not exploiting the parallelism in the dataloader. Instead, we can do this:

```julia
dataloader = DataLoader(dataset; parallel=true, batchsize=12)  # from MLUtils.jl
gdev = gpu_device()

for (X, y) in gdev(dataloader)
    # ...
    # do some computation
    # ...
end
```

Here, `X` and `y` are on the gpu device `gdev` and the data transfer happens in the
worker processes. Additionally, it behaves similar to `CuIterator` from CUDA.jl and eagerly
frees the data after every iteration (this is device agnostic and works on all supported GPU
backends).

## Type Instabilities

!!! tip "Using Reactant?"

    If you are using Lux with Reactant, type-instabilities won't affect performance at all.
    You can safely ignore this section.

`Lux.jl` is integrated with `DispatchDoctor.jl` to catch type instabilities. You can easily
enable it by setting the `instability_check` preference. This will help you catch type
instabilities in your code. For more information on how to set preferences, check out
[`Lux.set_dispatch_doctor_preferences!`](@ref).

## Faster Primitives

!!! tip "Using Reactant?"

    If you are using Lux with Reactant, we will automatically use optimized versions of the
    functions at the compiler level. You can safely ignore this section.

Prefer to use deep learning primitives and their fused variants from `LuxLib.jl` instead of
`NNlib.jl`. Some of the alternatives are:

1. Replace `NNlib.batched_mul` with [`LuxLib.batched_matmul`](@ref).
2. Replace `NNlib.conv` with bias and activation with
   [`LuxLib.fused_conv_bias_activation`](@ref).
3. Replace `σ.(w * x .+ b)` with [`LuxLib.fused_dense_bias_activation`](@ref).
4. Replace uses of `σ.(x)` with [`LuxLib.fast_activation`](@ref) or
   [`LuxLib.fast_activation!!`](@ref) (the latter one is often faster).
5. Replace uses of `σ.(x .+ b)` with [`LuxLib.bias_activation`](@ref) or
   [`LuxLib.bias_activation!!`](@ref) (the latter one is often faster).

## Optional Dependencies for Performance

!!! tip "Using Reactant?"

    These dependencies are not needed for Reactant. You can safely ignore this section.

For faster performance on CPUs load the following packages:

1. `LoopVectorization.jl`
2. `Octavian.jl`

If these are available, we automatically use optimized versions of the layers. Though there
are cases where this might be an issue (see
[#980](https://github.com/LuxDL/Lux.jl/issues/980) and
[disabling loop vectorization](@ref disable_loop_vectorization)).
