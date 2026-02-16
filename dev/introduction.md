---
url: /dev/introduction.md
---
# Introduction {#getting-started}

## Installation {#Installation}

Install [Julia v1.10 or above](https://julialang.org/downloads/). Lux.jl is available through the Julia package manager. You can enter it by pressing `]` in the REPL and then typing `add Lux`. Alternatively, you can also do

```julia
import Pkg
Pkg.add("Lux")
```

::: tip Update to v1

If you are using a pre-v1 version of Lux.jl, please see the [Updating to v1 section](/introduction/updating_to_v1#updating-to-v1) for instructions on how to update.

:::

## Quickstart {#Quickstart}

::: tip Pre-Requisites

You need to install `Optimisers`, `Reactant` and `Enzyme` if not done already. `Pkg.add(["Optimisers", "Enzyme", "Reactant"])`

:::

```julia
using Lux, Random, Optimisers, Enzyme, Reactant
```

```ansi
[34m[1m┌ [22m[39m[34m[1mDebug: [22m[39mPersistent compilation cache enabled. Using base directory: /home/runner/.julia/scratchspaces/3c362404-f566-11ee-1572-e11a4b42c853/xla_persistent_cache_0_0_326
[34m[1m└ [22m[39m[90m@ Reactant.PersistentCompileCache ~/.julia/packages/Reactant/k9g9w/src/PersistentCompileCache.jl:28[39m
[34m[1m┌ [22m[39m[34m[1mDebug: [22m[39mKernel cache enabled: false
[34m[1m└ [22m[39m[90m@ Reactant.PersistentCompileCache ~/.julia/packages/Reactant/k9g9w/src/PersistentCompileCache.jl:33[39m
[34m[1m┌ [22m[39m[34m[1mDebug: [22m[39mAutotune cache enabled: true
[34m[1m└ [22m[39m[90m@ Reactant.PersistentCompileCache ~/.julia/packages/Reactant/k9g9w/src/PersistentCompileCache.jl:38[39m
[34m[1m┌ [22m[39m[34m[1mDebug: [22m[39mREACTANT_XLA_RUNTIME:
[34m[1m│ [22m[39m  REACTANT_XLA_RUNTIME = "PJRT"
[34m[1m└ [22m[39m[90m@ Reactant.XLA ~/.julia/packages/Reactant/k9g9w/src/xla/XLA.jl:199[39m
```

We take randomness very seriously

```julia
# Seeding
rng = Random.default_rng()
Random.seed!(rng, 0)
```

```ansi
Random.TaskLocalRNG()
```

Build the model

```julia
# Construct the layer
model = Chain(Dense(128, 256, tanh), Chain(Dense(256, 256, tanh), Dense(256, 10)))
```

```ansi
Chain(
    layer_1 = Dense(128 => 256, tanh),            [90m# 33_024 parameters[39m
    layer_2 = Chain(
        layer_1 = Dense(256 => 256, tanh),        [90m# 65_792 parameters[39m
        layer_2 = Dense(256 => 10),               [90m# 2_570 parameters[39m
    ),
) [90m        # Total: [39m101_386 parameters,
[90m          #        plus [39m0 states.
```

Models don't hold parameters and states so initialize them. From there on, we can just use our standard AD and Optimisers API. However, here we will show how to use Lux's Training API that provides an uniform API over all supported AD systems.

```julia
# Get the device determined by Lux
dev = reactant_device()

# Parameter and State Variables
ps, st = Lux.setup(rng, model) |> dev

# Dummy Input
x = rand(rng, Float32, 128, 2) |> dev

# Run the model
## We need to use @jit to compile and run the model with Reactant
y, st = @jit Lux.apply(model, x, ps, st)

## For best performance, first compile the model with Reactant and then run it
apply_compiled = @compile Lux.apply(model, x, ps, st)
apply_compiled(model, x, ps, st)

# Gradients
## First construct a TrainState
train_state = Training.TrainState(model, ps, st, Adam(0.0001f0))

## We can compute the gradients using Training.compute_gradients
## TrainState handles compilation internally
gs, loss, stats, train_state = Lux.Training.compute_gradients(
    AutoEnzyme(),
    MSELoss(),
    (x, dev(rand(rng, Float32, 10, 2))),
    train_state
)

## Optimization
train_state = Training.apply_gradients!(train_state, gs) # or Training.apply_gradients (no `!` at the end)

# Both these steps can be combined into a single call (preferred approach)
gs, loss, stats, train_state = Training.single_train_step!(
    AutoEnzyme(),
    MSELoss(),
    (x, dev(rand(rng, Float32, 10, 2))),
    train_state
)
```

```ansi
((layer_1 = (weight = Reactant.ConcretePJRTArray{Float32, 2, 1}(Float32[-0.014252576 -0.01007682 … -0.0066116317 -0.012908219; -0.0066900905 0.001967808 … -0.003597645 -0.0010663884; … ; -0.013411599 -0.009727624 … -0.0062034056 -0.012329484; -0.021803929 -0.008291057 … -0.010640304 -0.014436427]), bias = Reactant.ConcretePJRTArray{Float32, 1, 1}(Float32[-0.016153783, -0.0064898166, 0.0040475307, 0.048172455, 0.018123386, -0.026255043, 0.031807806, -0.00242692, 0.0047841636, 0.030119237  …  -0.018572997, 0.002068058, 0.009790083, 0.018018095, -0.01424905, -0.060925536, -0.016411005, -0.014629741, -0.015240658, -0.023550106])), layer_2 = (layer_1 = (weight = Reactant.ConcretePJRTArray{Float32, 2, 1}(Float32[-2.1789727f-5 -3.8227197f-5 … 2.2419797f-6 -6.324912f-5; 0.008564907 0.006512975 … -0.006969174 0.009865682; … ; -0.021595063 -0.021455316 … 0.013971797 -0.033741944; -0.023399103 -0.018417688 … 0.018593067 -0.028052688]), bias = Reactant.ConcretePJRTArray{Float32, 1, 1}(Float32[-8.567904f-5, 0.015442661, -0.012886962, 0.052984916, 0.000940611, -0.013805261, -0.025449416, 0.012329709, 0.09247952, -0.038467444  …  0.02124564, -0.012647957, -0.00045393303, -0.026561411, -0.025026277, -0.044307135, 0.033703856, 0.02700105, -0.049719024, -0.043526463])), layer_2 = (weight = Reactant.ConcretePJRTArray{Float32, 2, 1}(Float32[-0.041988406 0.0035988672 … 0.012094588 -0.006939848; -0.06445931 0.02036321 … 0.044551518 0.008209803; … ; 0.24316937 -0.04692524 … -0.11571932 0.0070323865; 0.027664784 -0.008376195 … -0.018484466 -0.0030616056]), bias = Reactant.ConcretePJRTArray{Float32, 1, 1}(Float32[0.04771936, 0.079492785, -0.063023046, 0.011547155, -0.4143036, -0.118838035, -0.16815118, -0.2632838, -0.2873201, -0.0339642])))), Reactant.ConcretePJRTNumber{Float32, 1}(1.0014447f0), NamedTuple(), Lux.Training.TrainState{Lux.Training.TrainingBackendCache{Lux.Training.ReactantBackend{Static.True, Missing, Nothing, AutoEnzyme{Nothing, Nothing}}, Static.False, @NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}}}, @NamedTuple{compiled_gradient_function::Reactant.Compiler.Thunk{typeof(ReactantExt.compute_gradients_internal), Symbol("##compute_gradients_internal_reactant#487"), false, Tuple{GenericLossFunction{typeof(Lux.LossFunctionImpl.l2_distance_loss), typeof(Statistics.mean)}, Chain{@NamedTuple{layer_1::Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Chain{@NamedTuple{layer_1::Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Dense{typeof(identity), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing}}, Nothing}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}}, @NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}}}, @NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}}}}, Reactant.XLA.PJRT.LoadedExecutable, Reactant.XLA.PJRT.Device, Reactant.XLA.PJRT.Client, Tuple{}, Vector{Bool}}, update_function::Reactant.Compiler.Thunk{typeof(Optimisers.update!), Symbol("##update!_reactant#740"), false, Tuple{@NamedTuple{layer_1::@NamedTuple{weight::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}, bias::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 1, 1}, Reactant.ConcretePJRTArray{Float32, 1, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}, bias::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 1, 1}, Reactant.ConcretePJRTArray{Float32, 1, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}}, layer_2::@NamedTuple{weight::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}, bias::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 1, 1}, Reactant.ConcretePJRTArray{Float32, 1, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}}}}, @NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}}}, @NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}}}}, Reactant.XLA.PJRT.LoadedExecutable, Reactant.XLA.PJRT.Device, Reactant.XLA.PJRT.Client, Tuple{}, Vector{Bool}}, compiled_grad_and_step_function::Reactant.Compiler.Thunk{typeof(ReactantExt.compute_gradients_internal_and_step!), Symbol("##compute_gradients_internal_and_step!_reactant#1079"), false, Tuple{GenericLossFunction{typeof(Lux.LossFunctionImpl.l2_distance_loss), typeof(Statistics.mean)}, Chain{@NamedTuple{layer_1::Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Chain{@NamedTuple{layer_1::Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Dense{typeof(identity), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing}}, Nothing}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}}, @NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}}}, @NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}}}, @NamedTuple{layer_1::@NamedTuple{weight::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}, bias::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 1, 1}, Reactant.ConcretePJRTArray{Float32, 1, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}, bias::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 1, 1}, Reactant.ConcretePJRTArray{Float32, 1, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}}, layer_2::@NamedTuple{weight::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}, bias::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 1, 1}, Reactant.ConcretePJRTArray{Float32, 1, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}}}}, @NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}}}, Bool}, Reactant.XLA.PJRT.LoadedExecutable, Reactant.XLA.PJRT.Device, Reactant.XLA.PJRT.Client, Tuple{}, Vector{Bool}}, is_sharded::Bool}}, GenericLossFunction{typeof(Lux.LossFunctionImpl.l2_distance_loss), typeof(Statistics.mean)}, Nothing, Chain{@NamedTuple{layer_1::Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Chain{@NamedTuple{layer_1::Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Dense{typeof(identity), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing}}, Nothing}, @NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}}}, @NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}}}, Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, @NamedTuple{layer_1::@NamedTuple{weight::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}, bias::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 1, 1}, Reactant.ConcretePJRTArray{Float32, 1, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}, bias::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 1, 1}, Reactant.ConcretePJRTArray{Float32, 1, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}}, layer_2::@NamedTuple{weight::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}, bias::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 1, 1}, Reactant.ConcretePJRTArray{Float32, 1, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}}}}}(Lux.Training.TrainingBackendCache{Lux.Training.ReactantBackend{Static.True, Missing, Nothing, AutoEnzyme{Nothing, Nothing}}, Static.False, @NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}}}, @NamedTuple{compiled_gradient_function::Reactant.Compiler.Thunk{typeof(ReactantExt.compute_gradients_internal), Symbol("##compute_gradients_internal_reactant#487"), false, Tuple{GenericLossFunction{typeof(Lux.LossFunctionImpl.l2_distance_loss), typeof(Statistics.mean)}, Chain{@NamedTuple{layer_1::Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Chain{@NamedTuple{layer_1::Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Dense{typeof(identity), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing}}, Nothing}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}}, @NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}}}, @NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}}}}, Reactant.XLA.PJRT.LoadedExecutable, Reactant.XLA.PJRT.Device, Reactant.XLA.PJRT.Client, Tuple{}, Vector{Bool}}, update_function::Reactant.Compiler.Thunk{typeof(Optimisers.update!), Symbol("##update!_reactant#740"), false, Tuple{@NamedTuple{layer_1::@NamedTuple{weight::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}, bias::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 1, 1}, Reactant.ConcretePJRTArray{Float32, 1, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}, bias::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 1, 1}, Reactant.ConcretePJRTArray{Float32, 1, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}}, layer_2::@NamedTuple{weight::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}, bias::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 1, 1}, Reactant.ConcretePJRTArray{Float32, 1, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}}}}, @NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}}}, @NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}}}}, Reactant.XLA.PJRT.LoadedExecutable, Reactant.XLA.PJRT.Device, Reactant.XLA.PJRT.Client, Tuple{}, Vector{Bool}}, compiled_grad_and_step_function::Reactant.Compiler.Thunk{typeof(ReactantExt.compute_gradients_internal_and_step!), Symbol("##compute_gradients_internal_and_step!_reactant#1079"), false, Tuple{GenericLossFunction{typeof(Lux.LossFunctionImpl.l2_distance_loss), typeof(Statistics.mean)}, Chain{@NamedTuple{layer_1::Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Chain{@NamedTuple{layer_1::Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Dense{typeof(identity), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing}}, Nothing}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}}, @NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}}}, @NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}}}, @NamedTuple{layer_1::@NamedTuple{weight::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}, bias::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 1, 1}, Reactant.ConcretePJRTArray{Float32, 1, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}, bias::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 1, 1}, Reactant.ConcretePJRTArray{Float32, 1, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}}, layer_2::@NamedTuple{weight::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}, bias::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 1, 1}, Reactant.ConcretePJRTArray{Float32, 1, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}}}}, @NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}}}, Bool}, Reactant.XLA.PJRT.LoadedExecutable, Reactant.XLA.PJRT.Device, Reactant.XLA.PJRT.Client, Tuple{}, Vector{Bool}}, is_sharded::Bool}}(Lux.Training.ReactantBackend{Static.True, Missing, Nothing, AutoEnzyme{Nothing, Nothing}}(static(true), missing, nothing, AutoEnzyme()), static(false), (layer_1 = (weight = Reactant.ConcretePJRTArray{Float32, 2, 1}(Float32[-0.014252576 -0.01007682 … -0.0066116317 -0.012908219; -0.0066900905 0.001967808 … -0.003597645 -0.0010663884; … ; -0.013411599 -0.009727624 … -0.0062034056 -0.012329484; -0.021803929 -0.008291057 … -0.010640304 -0.014436427]), bias = Reactant.ConcretePJRTArray{Float32, 1, 1}(Float32[-0.016153783, -0.0064898166, 0.0040475307, 0.048172455, 0.018123386, -0.026255043, 0.031807806, -0.00242692, 0.0047841636, 0.030119237  …  -0.018572997, 0.002068058, 0.009790083, 0.018018095, -0.01424905, -0.060925536, -0.016411005, -0.014629741, -0.015240658, -0.023550106])), layer_2 = (layer_1 = (weight = Reactant.ConcretePJRTArray{Float32, 2, 1}(Float32[-2.1789727f-5 -3.8227197f-5 … 2.2419797f-6 -6.324912f-5; 0.008564907 0.006512975 … -0.006969174 0.009865682; … ; -0.021595063 -0.021455316 … 0.013971797 -0.033741944; -0.023399103 -0.018417688 … 0.018593067 -0.028052688]), bias = Reactant.ConcretePJRTArray{Float32, 1, 1}(Float32[-8.567904f-5, 0.015442661, -0.012886962, 0.052984916, 0.000940611, -0.013805261, -0.025449416, 0.012329709, 0.09247952, -0.038467444  …  0.02124564, -0.012647957, -0.00045393303, -0.026561411, -0.025026277, -0.044307135, 0.033703856, 0.02700105, -0.049719024, -0.043526463])), layer_2 = (weight = Reactant.ConcretePJRTArray{Float32, 2, 1}(Float32[-0.041988406 0.0035988672 … 0.012094588 -0.006939848; -0.06445931 0.02036321 … 0.044551518 0.008209803; … ; 0.24316937 -0.04692524 … -0.11571932 0.0070323865; 0.027664784 -0.008376195 … -0.018484466 -0.0030616056]), bias = Reactant.ConcretePJRTArray{Float32, 1, 1}(Float32[0.04771936, 0.079492785, -0.063023046, 0.011547155, -0.4143036, -0.118838035, -0.16815118, -0.2632838, -0.2873201, -0.0339642])))), (compiled_gradient_function = Reactant compiled function compute_gradients_internal (with tag ##compute_gradients_internal_reactant#487), update_function = Reactant compiled function update! (with tag ##update!_reactant#740), compiled_grad_and_step_function = Reactant compiled function compute_gradients_internal_and_step! (with tag ##compute_gradients_internal_and_step!_reactant#1079), is_sharded = false)), GenericLossFunction{typeof(Lux.LossFunctionImpl.l2_distance_loss), typeof(Statistics.mean)}(Lux.LossFunctionImpl.l2_distance_loss, Statistics.mean), nothing, Chain{@NamedTuple{layer_1::Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Chain{@NamedTuple{layer_1::Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Dense{typeof(identity), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing}}, Nothing}((layer_1 = Dense(128 => 256, tanh), layer_2 = Chain{@NamedTuple{layer_1::Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Dense{typeof(identity), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing}((layer_1 = Dense(256 => 256, tanh), layer_2 = Dense(256 => 10)), nothing)), nothing), (layer_1 = (weight = Reactant.ConcretePJRTArray{Float32, 2, 1}(Float32[-0.2252784 0.22416925 … 0.1999151 -0.018334232; -0.022642436 0.15450513 … -0.06492576 0.18159999; … ; 0.038053643 -0.071250185 … -0.033061065 0.039139703; -0.18772855 -0.0965359 … -0.18063419 0.019627318]), bias = Reactant.ConcretePJRTArray{Float32, 1, 1}(Float32[0.031100241, -0.059878908, 0.08487941, 2.0693342f-6, -0.06550352, -0.08487986, -0.026524307, 0.06386332, 0.042082705, 0.02731392  …  -0.060524136, 0.0346705, -0.02837894, 0.06748107, 0.0027473217, -0.06903143, 0.006823757, 0.014109333, -0.029280972, 0.012123728])), layer_2 = (layer_1 = (weight = Reactant.ConcretePJRTArray{Float32, 2, 1}(Float32[0.120222874 -0.06340912 … -0.0006328311 -0.08487568; 0.05999857 0.08031667 … 0.10135459 -0.008547246; … ; -0.070574954 -0.12529366 … -0.112988465 0.03760839; 0.15816367 -0.12347525 … 0.01897169 0.1261452]), bias = Reactant.ConcretePJRTArray{Float32, 1, 1}(Float32[-0.047726065, -0.0076126982, 0.05891254, -0.05468241, 0.054435007, -0.005947113, -0.03260495, 0.025213236, 0.0024704752, 0.019756647  …  -0.038096465, -0.029946823, 0.021725217, -0.004842341, 0.0018720245, -0.029938111, 0.008292873, 0.01365754, 0.053042553, -0.02033623])), layer_2 = (weight = Reactant.ConcretePJRTArray{Float32, 2, 1}(Float32[-0.10451723 -0.027075663 … 0.018401911 0.06574812; -0.09850936 -0.10326829 … -0.021818167 0.047744956; … ; -0.061319478 -0.06494065 … 0.05045609 0.06725229; -0.025141204 0.094486676 … 0.02057817 -0.09107423]), bias = Reactant.ConcretePJRTArray{Float32, 1, 1}(Float32[-0.03402, 0.055178143, -0.036147784, 0.03546726, -0.026433855, -0.037797134, -0.02685109, -0.05179471, 0.05912261, -0.05172773])))), (layer_1 = NamedTuple(), layer_2 = (layer_1 = NamedTuple(), layer_2 = NamedTuple())), ReactantOptimiser(Adam(eta=Reactant.ConcretePJRTNumber{Float32, 1}(0.0001f0), beta=(Reactant.ConcretePJRTNumber{Float64, 1}(0.9), Reactant.ConcretePJRTNumber{Float64, 1}(0.999)), epsilon=Reactant.ConcretePJRTNumber{Float64, 1}(1.0e-8))), (layer_1 = (weight = [32mLeaf(ReactantOptimiser(Adam(eta=Reactant.ConcretePJRTNumber{Float32, 1}(0.0001f0), beta=(Reactant.ConcretePJRTNumber{Float64, 1}(0.9), Reactant.ConcretePJRTNumber{Float64, 1}(0.999)), epsilon=Reactant.ConcretePJRTNumber{Float64, 1}(1.0e-8))), [39m(Reactant.ConcretePJRTArray{Float32, 2, 1}(Float32[-0.00127828 -0.00108819 … -0.000579374 -0.00129518; -0.00129653 0.000363266 … -0.000695885 -0.000220152; … ; -0.00226619 -0.00174052 … -0.00104106 -0.00215551; -0.00363046 -0.00162021 … -0.00175398 -0.00258242]), Reactant.ConcretePJRTArray{Float32, 2, 1}(Float32[2.05798f-7 1.0234f-7 … 4.45381f-8 1.66622f-7; 9.3323f-8 7.29063f-9 … 2.68765f-8 2.72633f-9; … ; 2.85402f-7 1.67323f-7 … 6.03123f-8 2.56985f-7; 7.34734f-7 1.45927f-7 … 1.71924f-7 3.68345f-7]), (Reactant.ConcretePJRTNumber{Float32, 1}(0.729), Reactant.ConcretePJRTNumber{Float32, 1}(0.997003)))[32m)[39m, bias = [32mLeaf(ReactantOptimiser(Adam(eta=Reactant.ConcretePJRTNumber{Float32, 1}(0.0001f0), beta=(Reactant.ConcretePJRTNumber{Float64, 1}(0.9), Reactant.ConcretePJRTNumber{Float64, 1}(0.999)), epsilon=Reactant.ConcretePJRTNumber{Float64, 1}(1.0e-8))), [39m(Reactant.ConcretePJRTArray{Float32, 1, 1}(Float32[-0.00147888, -0.00126067, -0.00024512, 0.0103003, 0.00511794, -0.00677713, 0.00692629, -0.000736556, 0.000429644, 0.00477295  …  -0.00376831, 0.000957916, 0.000537737, 0.00326793, -0.00284112, -0.0107958, -0.00270653, -0.00281344, -0.00259105, -0.00396031]), Reactant.ConcretePJRTArray{Float32, 1, 1}(Float32[2.63239f-7, 8.82635f-8, 6.84696f-8, 6.02842f-6, 1.6761f-6, 2.81506f-6, 2.74192f-6, 3.59707f-8, 2.31813f-8, 1.28963f-6  …  7.95356f-7, 7.38564f-8, 1.1986f-7, 5.89749f-7, 4.50396f-7, 6.44006f-6, 4.09317f-7, 4.38955f-7, 3.72682f-7, 8.72425f-7]), (Reactant.ConcretePJRTNumber{Float32, 1}(0.729), Reactant.ConcretePJRTNumber{Float32, 1}(0.997003)))[32m)[39m), layer_2 = (layer_1 = (weight = [32mLeaf(ReactantOptimiser(Adam(eta=Reactant.ConcretePJRTNumber{Float32, 1}(0.0001f0), beta=(Reactant.ConcretePJRTNumber{Float64, 1}(0.9), Reactant.ConcretePJRTNumber{Float64, 1}(0.999)), epsilon=Reactant.ConcretePJRTNumber{Float64, 1}(1.0e-8))), [39m(Reactant.ConcretePJRTArray{Float32, 2, 1}(Float32[-1.92071f-5 -1.98882f-6 … 2.45465f-5 9.11111f-8; 0.0017717 0.00124793 … -0.00150786 0.00187327; … ; -0.00373359 -0.0036783 … 0.00243024 -0.00579673; -0.00454264 -0.00359967 … 0.00358124 -0.0055097]), Reactant.ConcretePJRTArray{Float32, 2, 1}(Float32[3.62358f-11 1.87609f-12 … 7.29646f-11 9.0774f-12; 1.76661f-7 8.63204f-8 … 1.29674f-7 1.943f-7; … ; 7.71925f-7 7.50078f-7 … 3.2683f-7 1.8623f-6; 1.14592f-6 7.20328f-7 … 7.11382f-7 1.68898f-6]), (Reactant.ConcretePJRTNumber{Float32, 1}(0.729), Reactant.ConcretePJRTNumber{Float32, 1}(0.997003)))[32m)[39m, bias = [32mLeaf(ReactantOptimiser(Adam(eta=Reactant.ConcretePJRTNumber{Float32, 1}(0.0001f0), beta=(Reactant.ConcretePJRTNumber{Float64, 1}(0.9), Reactant.ConcretePJRTNumber{Float64, 1}(0.999)), epsilon=Reactant.ConcretePJRTNumber{Float64, 1}(1.0e-8))), [39m(Reactant.ConcretePJRTArray{Float32, 1, 1}(Float32[-7.6764f-6, 0.002996, -0.00215631, 0.0108377, 0.000171212, -0.00324612, -0.00506657, 0.00160721, 0.0186094, -0.00646918  …  0.00417523, -0.0043, -0.000207296, -0.0046866, -0.00398426, -0.00686918, 0.00543797, 0.00308305, -0.00856162, -0.00854114]), Reactant.ConcretePJRTArray{Float32, 1, 1}(Float32[7.43883f-12, 4.98398f-7, 2.5891f-7, 6.59154f-6, 1.61884f-9, 6.19829f-7, 1.43188f-6, 1.69292f-7, 1.93607f-5, 2.32789f-6  …  9.70009f-7, 1.29615f-6, 3.43886f-9, 1.21397f-6, 8.97047f-7, 2.69644f-6, 1.66317f-6, 7.47133f-7, 4.06121f-6, 4.05819f-6]), (Reactant.ConcretePJRTNumber{Float32, 1}(0.729), Reactant.ConcretePJRTNumber{Float32, 1}(0.997003)))[32m)[39m), layer_2 = (weight = [32mLeaf(ReactantOptimiser(Adam(eta=Reactant.ConcretePJRTNumber{Float32, 1}(0.0001f0), beta=(Reactant.ConcretePJRTNumber{Float64, 1}(0.9), Reactant.ConcretePJRTNumber{Float64, 1}(0.999)), epsilon=Reactant.ConcretePJRTNumber{Float64, 1}(1.0e-8))), [39m(Reactant.ConcretePJRTArray{Float32, 2, 1}(Float32[-0.00948004 0.00171439 … 0.00412935 -0.000683905; -0.00407593 0.0025052 … 0.00502547 0.00215895; … ; 0.0472378 -0.00898661 … -0.0214768 0.00263519; 0.0115309 -0.00133463 … -0.00363606 0.00186014]), Reactant.ConcretePJRTArray{Float32, 2, 1}(Float32[5.20286f-6 2.39227f-7 … 1.19777f-6 4.81734f-8; 4.84769f-6 4.41769f-7 … 2.02493f-6 2.88184f-7; … ; 0.000123925 4.47608f-6 … 2.54904f-5 5.09783f-7; 1.02391f-5 1.00626f-7 … 7.35784f-7 5.8815f-7]), (Reactant.ConcretePJRTNumber{Float32, 1}(0.729), Reactant.ConcretePJRTNumber{Float32, 1}(0.997003)))[32m)[39m, bias = [32mLeaf(ReactantOptimiser(Adam(eta=Reactant.ConcretePJRTNumber{Float32, 1}(0.0001f0), beta=(Reactant.ConcretePJRTNumber{Float64, 1}(0.9), Reactant.ConcretePJRTNumber{Float64, 1}(0.999)), epsilon=Reactant.ConcretePJRTNumber{Float64, 1}(1.0e-8))), [39m(Reactant.ConcretePJRTArray{Float32, 1, 1}(Float32[0.0111157, 0.00553539, -0.0061385, 0.00853476, -0.0791731, -0.0292157, -0.0252973, -0.0456424, -0.0556272, -0.0132202]), Reactant.ConcretePJRTArray{Float32, 1, 1}(Float32[7.24034f-6, 7.03766f-6, 3.97516f-6, 6.8506f-6, 0.000347333, 5.11703f-5, 3.71479f-5, 0.000115324, 0.000171764, 1.30559f-5]), (Reactant.ConcretePJRTNumber{Float32, 1}(0.729), Reactant.ConcretePJRTNumber{Float32, 1}(0.997003)))[32m)[39m))), 2))
```

## Defining Custom Layers {#Defining-Custom-Layers}

We can train our model using the above code, but let's go ahead and see how to use Reactant. Reactant is a julia frontend that generates MLIR and then compiles it using XLA (after running fancy optimizations). It is the current recommended way to train large models in Lux. For more details on using Reactant, see the [manual](/manual/compiling_lux_models#reactant-compilation).

```julia
using Lux, Random, Optimisers, Reactant, Enzyme
using Printf # For pretty printing

dev = reactant_device()
```

```ansi
(::ReactantDevice{Missing, Missing, Missing, Missing, Union{}}) (generic function with 1 method)
```

We will define a custom MLP using the `@compact` macro. The macro takes in a list of parameters, layers and states, and a function defining the forward pass of the neural network.

```julia
n_in = 1
n_out = 1
nlayers = 3

model = @compact(
    w1=Dense(n_in => 32),
    w2=[Dense(32 => 32) for i in 1:nlayers],
    w3=Dense(32 => n_out),
    act=relu
) do x
    embed = act(w1(x))
    for w in w2
        embed = act(w(embed))
    end
    out = w3(embed)
    @return out
end
```

```ansi
@compact(
    w1 = Dense(1 => 32),                          [90m# 64 parameters[39m
    w2 = NamedTuple(
        (1-3) = Dense(32 => 32),                  [90m# 3_168 (1_056 x 3) parameters[39m
    ),
    w3 = Dense(32 => 1),                          [90m# 33 parameters[39m
    act = relu,
) do x 
    embed = act(w1(x))
    for w = w2
        embed = act(w(embed))
    end
    out = w3(embed)
    return out
end[90m       # Total: [39m3_265 parameters,
[90m          #        plus [39m1 states.
```

We can initialize the model and train it with the same code as before!

```julia
rng = Random.default_rng()
Random.seed!(rng, 0)

ps, st = Lux.setup(rng, model) |> dev

x = rand(rng, Float32, n_in, 32) |> dev

@jit model(x, ps, st)  # 1×32 Matrix and updated state as output.

x_data = reshape(collect(-2.0f0:0.1f0:2.0f0), 1, :)
y_data = 2 .* x_data .- x_data .^ 3
x_data, y_data = dev(x_data), dev(y_data)

function train_model!(model, ps, st, x_data, y_data, num_epochs=1000)
    train_state = Lux.Training.TrainState(model, ps, st, Adam(0.001f0))

    for iter in 1:num_epochs
        _, loss, _, train_state = Lux.Training.single_train_step!(
            AutoEnzyme(), MSELoss(),
            (x_data, y_data), train_state
        )
        if iter == 1 || iter % 100 == 0 || iter == num_epochs
            @printf "Iteration: %04d \t Loss: %10.9g\n" iter loss
        end
    end

    return model, train_state.parameters, train_state.states
end

train_model!(model, ps, st, x_data, y_data)
```

```ansi
Iteration: 0001 	 Loss: 2.08085132
Iteration: 0100 	 Loss: 0.138668075
Iteration: 0200 	 Loss: 0.00399366021
Iteration: 0300 	 Loss: 0.00102391699
Iteration: 0400 	 Loss: 0.000365888263
Iteration: 0500 	 Loss: 0.000234172869
Iteration: 0600 	 Loss: 0.000158297669
Iteration: 0700 	 Loss: 0.000124764629
Iteration: 0800 	 Loss: 0.000425997976
Iteration: 0900 	 Loss: 0.00031550636
Iteration: 1000 	 Loss: 7.92308856e-05
```

::: tip Training with Optimization.jl

If you are coming from the SciML ecosystem and want to use Optimization.jl, please refer to the [Optimization.jl Tutorial](/tutorials/beginner/5_OptimizationIntegration#Optimization-Lux-Tutorial).

:::

## Additional Packages {#Additional-Packages}

`LuxDL` hosts various packages that provide additional functionality for Lux.jl. All packages mentioned in this documentation are available via the Julia General Registry.

You can install all those packages via `import Pkg; Pkg.add(<package name>)`.

## Reactant & XLA (CPU/GPU/TPU) Support {#Reactant-and-XLA-CPU/GPU/TPU-Support}

Lux.jl supports XLA compilation for CPU, GPU, and TPU using [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl).

## Native Julia GPU Support {#Native-Julia-GPU-Support}

::: warning Warning

Using accelerators via Reactant and XLA is the preferred way to use GPUs with Lux.jl.

:::

::: danger AMD GPU Users

For AMD GPUs, we **strongly recommend using Reactant** instead of native AMDGPU.jl support. Native AMDGPU.jl support is experimental with known issues including deadlocks. Reactant provides superior performance and reliability for AMD hardware.

:::

GPU Support for Lux.jl requires loading additional packages:

* [`LuxCUDA.jl`](https://github.com/LuxDL/LuxCUDA.jl) for CUDA support.

* [`AMDGPU.jl`](https://github.com/JuliaGPU/AMDGPU.jl) for AMDGPU support (not recommended, use Reactant instead).

* [`Metal.jl`](https://github.com/JuliaGPU/Metal.jl) for Apple Metal support.

* [`oneAPI.jl`](https://github.com/JuliaGPU/oneAPI.jl) for oneAPI support.
