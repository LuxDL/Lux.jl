---
url: /v1/introduction.md
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
[34m[1m┌ [22m[39m[34m[1mDebug: [22m[39mPersistent compilation cache enabled. Using base directory: /home/runner/.julia/scratchspaces/3c362404-f566-11ee-1572-e11a4b42c853/xla_persistent_cache_0_0_447
[34m[1m└ [22m[39m[90m@ Reactant.PersistentCompileCache ~/.julia/packages/Reactant/BjNC7/src/PersistentCompileCache.jl:28[39m
[34m[1m┌ [22m[39m[34m[1mDebug: [22m[39mKernel cache enabled: false
[34m[1m└ [22m[39m[90m@ Reactant.PersistentCompileCache ~/.julia/packages/Reactant/BjNC7/src/PersistentCompileCache.jl:33[39m
[34m[1m┌ [22m[39m[34m[1mDebug: [22m[39mAutotune cache enabled: true
[34m[1m└ [22m[39m[90m@ Reactant.PersistentCompileCache ~/.julia/packages/Reactant/BjNC7/src/PersistentCompileCache.jl:38[39m
[34m[1m┌ [22m[39m[34m[1mDebug: [22m[39mRegistering Backend cpu with Priority 100.
[34m[1m└ [22m[39m[90m@ Reactant.Accelerators.Registration ~/.julia/packages/Reactant/BjNC7/src/accelerators/Registration.jl:39[39m
[34m[1m┌ [22m[39m[34m[1mDebug: [22m[39mUsing libReactantExtra from /home/runner/.julia/artifacts/e02636b6eda7b5cc0cf3cf4b84e363cb27530ecf/lib/libReactantExtra.so
[34m[1m└ [22m[39m[90m@ Reactant.XLA ~/.julia/packages/Reactant/BjNC7/src/xla/XLA.jl:302[39m
[34m[1m┌ [22m[39m[34m[1mDebug: [22m[39mREACTANT_XLA_RUNTIME:
[34m[1m│ [22m[39m  REACTANT_XLA_RUNTIME = "PJRT"
[34m[1m└ [22m[39m[90m@ Reactant.XLA ~/.julia/packages/Reactant/BjNC7/src/xla/XLA.jl:341[39m
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
((layer_1 = (weight = Reactant.ConcretePJRTArray{Float32, 2, 1}(Float32[-0.014252574 -0.010076822 … -0.0066116313 -0.01290822; -0.0066900887 0.00196781 … -0.0035976442 -0.0010663861; … ; -0.013411606 -0.00972764 … -0.006203408 -0.012329499; -0.021803938 -0.008291059 … -0.010640309 -0.014436432]), bias = Reactant.ConcretePJRTArray{Float32, 1, 1}(Float32[-0.016153783, -0.0064898143, 0.0040475302, 0.048172448, 0.018123366, -0.026255049, 0.0318078, -0.0024269335, 0.004784157, 0.030119218  …  -0.01857301, 0.0020680542, 0.009790082, 0.018018095, -0.01424905, -0.060925532, -0.016411027, -0.014629744, -0.015240667, -0.023550117])), layer_2 = (layer_1 = (weight = Reactant.ConcretePJRTArray{Float32, 2, 1}(Float32[-2.1792433f-5 -3.8228773f-5 … 2.2445429f-6 -6.325135f-5; 0.008564905 0.0065129735 … -0.0069691734 0.00986568; … ; -0.021595046 -0.021455316 … 0.0139717795 -0.03374194; -0.023399094 -0.018417684 … 0.01859306 -0.028052678]), bias = Reactant.ConcretePJRTArray{Float32, 1, 1}(Float32[-8.568286f-5, 0.015442656, -0.012886958, 0.052984923, 0.0009406151, -0.013805272, -0.025449418, 0.012329705, 0.09247953, -0.038467433  …  0.02124565, -0.012647951, -0.00045393186, -0.026561417, -0.02502627, -0.044307124, 0.033703864, 0.027001038, -0.049719013, -0.04352645])), layer_2 = (weight = Reactant.ConcretePJRTArray{Float32, 2, 1}(Float32[-0.041988432 0.0035988803 … 0.012094625 -0.006939831; -0.06445934 0.02036322 … 0.044551555 0.008209848; … ; 0.24316931 -0.046925254 … -0.11571939 0.007032255; 0.027664827 -0.008376193 … -0.018484475 -0.0030616089]), bias = Reactant.ConcretePJRTArray{Float32, 1, 1}(Float32[0.047719397, 0.07949282, -0.06302307, 0.011547143, -0.41430357, -0.11883809, -0.16815121, -0.26328388, -0.28732005, -0.03396425])))), Reactant.ConcretePJRTNumber{Float32, 1}(1.0014447f0), NamedTuple(), Lux.Training.TrainState{Lux.Training.TrainingBackendCache{Lux.Training.ReactantBackend{Static.True, Missing, Nothing, AutoEnzyme{Nothing, Nothing}}, Static.False, @NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}}}, @NamedTuple{compiled_gradient_function::Reactant.Compiler.Thunk{typeof(ReactantExt.compute_gradients_internal), Symbol("##compute_gradients_internal_reactant#290"), false, Tuple{GenericLossFunction{typeof(Lux.LossFunctionImpl.l2_distance_loss), typeof(Statistics.mean)}, Chain{@NamedTuple{layer_1::Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Chain{@NamedTuple{layer_1::Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Dense{typeof(identity), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing}}, Nothing}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}}, @NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}}}, @NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}}}}, Reactant.XLA.PJRT.LoadedExecutable, Reactant.XLA.PJRT.Device, Reactant.XLA.PJRT.Client, Tuple{}, Vector{Bool}}, update_function::Reactant.Compiler.Thunk{typeof(Optimisers.update!), Symbol("##update!_reactant#543"), false, Tuple{@NamedTuple{layer_1::@NamedTuple{weight::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}, bias::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 1, 1}, Reactant.ConcretePJRTArray{Float32, 1, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}, bias::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 1, 1}, Reactant.ConcretePJRTArray{Float32, 1, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}}, layer_2::@NamedTuple{weight::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}, bias::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 1, 1}, Reactant.ConcretePJRTArray{Float32, 1, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}}}}, @NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}}}, @NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}}}}, Reactant.XLA.PJRT.LoadedExecutable, Reactant.XLA.PJRT.Device, Reactant.XLA.PJRT.Client, Tuple{}, Vector{Bool}}, compiled_grad_and_step_function::Reactant.Compiler.Thunk{typeof(ReactantExt.compute_gradients_internal_and_step!), Symbol("##compute_gradients_internal_and_step!_reactant#864"), false, Tuple{GenericLossFunction{typeof(Lux.LossFunctionImpl.l2_distance_loss), typeof(Statistics.mean)}, Chain{@NamedTuple{layer_1::Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Chain{@NamedTuple{layer_1::Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Dense{typeof(identity), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing}}, Nothing}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}}, @NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}}}, @NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}}}, @NamedTuple{layer_1::@NamedTuple{weight::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}, bias::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 1, 1}, Reactant.ConcretePJRTArray{Float32, 1, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}, bias::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 1, 1}, Reactant.ConcretePJRTArray{Float32, 1, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}}, layer_2::@NamedTuple{weight::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}, bias::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 1, 1}, Reactant.ConcretePJRTArray{Float32, 1, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}}}}, @NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}}}, Bool}, Reactant.XLA.PJRT.LoadedExecutable, Reactant.XLA.PJRT.Device, Reactant.XLA.PJRT.Client, Tuple{}, Vector{Bool}}, is_sharded::Bool}}, GenericLossFunction{typeof(Lux.LossFunctionImpl.l2_distance_loss), typeof(Statistics.mean)}, Nothing, Chain{@NamedTuple{layer_1::Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Chain{@NamedTuple{layer_1::Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Dense{typeof(identity), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing}}, Nothing}, @NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}}}, @NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}}}, Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, @NamedTuple{layer_1::@NamedTuple{weight::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}, bias::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 1, 1}, Reactant.ConcretePJRTArray{Float32, 1, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}, bias::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 1, 1}, Reactant.ConcretePJRTArray{Float32, 1, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}}, layer_2::@NamedTuple{weight::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}, bias::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 1, 1}, Reactant.ConcretePJRTArray{Float32, 1, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}}}}}(Lux.Training.TrainingBackendCache{Lux.Training.ReactantBackend{Static.True, Missing, Nothing, AutoEnzyme{Nothing, Nothing}}, Static.False, @NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}}}, @NamedTuple{compiled_gradient_function::Reactant.Compiler.Thunk{typeof(ReactantExt.compute_gradients_internal), Symbol("##compute_gradients_internal_reactant#290"), false, Tuple{GenericLossFunction{typeof(Lux.LossFunctionImpl.l2_distance_loss), typeof(Statistics.mean)}, Chain{@NamedTuple{layer_1::Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Chain{@NamedTuple{layer_1::Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Dense{typeof(identity), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing}}, Nothing}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}}, @NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}}}, @NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}}}}, Reactant.XLA.PJRT.LoadedExecutable, Reactant.XLA.PJRT.Device, Reactant.XLA.PJRT.Client, Tuple{}, Vector{Bool}}, update_function::Reactant.Compiler.Thunk{typeof(Optimisers.update!), Symbol("##update!_reactant#543"), false, Tuple{@NamedTuple{layer_1::@NamedTuple{weight::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}, bias::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 1, 1}, Reactant.ConcretePJRTArray{Float32, 1, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}, bias::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 1, 1}, Reactant.ConcretePJRTArray{Float32, 1, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}}, layer_2::@NamedTuple{weight::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}, bias::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 1, 1}, Reactant.ConcretePJRTArray{Float32, 1, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}}}}, @NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}}}, @NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}}}}, Reactant.XLA.PJRT.LoadedExecutable, Reactant.XLA.PJRT.Device, Reactant.XLA.PJRT.Client, Tuple{}, Vector{Bool}}, compiled_grad_and_step_function::Reactant.Compiler.Thunk{typeof(ReactantExt.compute_gradients_internal_and_step!), Symbol("##compute_gradients_internal_and_step!_reactant#864"), false, Tuple{GenericLossFunction{typeof(Lux.LossFunctionImpl.l2_distance_loss), typeof(Statistics.mean)}, Chain{@NamedTuple{layer_1::Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Chain{@NamedTuple{layer_1::Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Dense{typeof(identity), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing}}, Nothing}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}}, @NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}}}, @NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}}}, @NamedTuple{layer_1::@NamedTuple{weight::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}, bias::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 1, 1}, Reactant.ConcretePJRTArray{Float32, 1, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}, bias::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 1, 1}, Reactant.ConcretePJRTArray{Float32, 1, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}}, layer_2::@NamedTuple{weight::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 2, 1}, Reactant.ConcretePJRTArray{Float32, 2, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}, bias::Optimisers.Leaf{Lux.ReactantCompatibleOptimisers.ReactantOptimiser{Adam{Reactant.ConcretePJRTNumber{Float32, 1}, Tuple{Reactant.ConcretePJRTNumber{Float64, 1}, Reactant.ConcretePJRTNumber{Float64, 1}}, Reactant.ConcretePJRTNumber{Float64, 1}}}, Tuple{Reactant.ConcretePJRTArray{Float32, 1, 1}, Reactant.ConcretePJRTArray{Float32, 1, 1}, Tuple{Reactant.ConcretePJRTNumber{Float32, 1}, Reactant.ConcretePJRTNumber{Float32, 1}}}}}}}, @NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{layer_1::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}, layer_2::@NamedTuple{weight::Reactant.ConcretePJRTArray{Float32, 2, 1}, bias::Reactant.ConcretePJRTArray{Float32, 1, 1}}}}, Bool}, Reactant.XLA.PJRT.LoadedExecutable, Reactant.XLA.PJRT.Device, Reactant.XLA.PJRT.Client, Tuple{}, Vector{Bool}}, is_sharded::Bool}}(Lux.Training.ReactantBackend{Static.True, Missing, Nothing, AutoEnzyme{Nothing, Nothing}}(static(true), missing, nothing, AutoEnzyme()), static(false), (layer_1 = (weight = Reactant.ConcretePJRTArray{Float32, 2, 1}(Float32[-0.014252574 -0.010076822 … -0.0066116313 -0.01290822; -0.0066900887 0.00196781 … -0.0035976442 -0.0010663861; … ; -0.013411606 -0.00972764 … -0.006203408 -0.012329499; -0.021803938 -0.008291059 … -0.010640309 -0.014436432]), bias = Reactant.ConcretePJRTArray{Float32, 1, 1}(Float32[-0.016153783, -0.0064898143, 0.0040475302, 0.048172448, 0.018123366, -0.026255049, 0.0318078, -0.0024269335, 0.004784157, 0.030119218  …  -0.01857301, 0.0020680542, 0.009790082, 0.018018095, -0.01424905, -0.060925532, -0.016411027, -0.014629744, -0.015240667, -0.023550117])), layer_2 = (layer_1 = (weight = Reactant.ConcretePJRTArray{Float32, 2, 1}(Float32[-2.1792433f-5 -3.8228773f-5 … 2.2445429f-6 -6.325135f-5; 0.008564905 0.0065129735 … -0.0069691734 0.00986568; … ; -0.021595046 -0.021455316 … 0.0139717795 -0.03374194; -0.023399094 -0.018417684 … 0.01859306 -0.028052678]), bias = Reactant.ConcretePJRTArray{Float32, 1, 1}(Float32[-8.568286f-5, 0.015442656, -0.012886958, 0.052984923, 0.0009406151, -0.013805272, -0.025449418, 0.012329705, 0.09247953, -0.038467433  …  0.02124565, -0.012647951, -0.00045393186, -0.026561417, -0.02502627, -0.044307124, 0.033703864, 0.027001038, -0.049719013, -0.04352645])), layer_2 = (weight = Reactant.ConcretePJRTArray{Float32, 2, 1}(Float32[-0.041988432 0.0035988803 … 0.012094625 -0.006939831; -0.06445934 0.02036322 … 0.044551555 0.008209848; … ; 0.24316931 -0.046925254 … -0.11571939 0.007032255; 0.027664827 -0.008376193 … -0.018484475 -0.0030616089]), bias = Reactant.ConcretePJRTArray{Float32, 1, 1}(Float32[0.047719397, 0.07949282, -0.06302307, 0.011547143, -0.41430357, -0.11883809, -0.16815121, -0.26328388, -0.28732005, -0.03396425])))), (compiled_gradient_function = Reactant compiled function compute_gradients_internal (with tag ##compute_gradients_internal_reactant#290), update_function = Reactant compiled function update! (with tag ##update!_reactant#543), compiled_grad_and_step_function = Reactant compiled function compute_gradients_internal_and_step! (with tag ##compute_gradients_internal_and_step!_reactant#864), is_sharded = false)), GenericLossFunction{typeof(Lux.LossFunctionImpl.l2_distance_loss), typeof(Statistics.mean)}(Lux.LossFunctionImpl.l2_distance_loss, Statistics.mean), nothing, Chain{@NamedTuple{layer_1::Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Chain{@NamedTuple{layer_1::Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Dense{typeof(identity), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing}}, Nothing}((layer_1 = Dense(128 => 256, tanh), layer_2 = Chain{@NamedTuple{layer_1::Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Dense{typeof(identity), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing}((layer_1 = Dense(256 => 256, tanh), layer_2 = Dense(256 => 10)), nothing)), nothing), (layer_1 = (weight = Reactant.ConcretePJRTArray{Float32, 2, 1}(Float32[-0.2252784 0.22416925 … 0.1999151 -0.018334232; -0.022642436 0.15450513 … -0.06492576 0.18159999; … ; 0.038053643 -0.071250185 … -0.033061065 0.039139703; -0.18772855 -0.0965359 … -0.18063419 0.019627318]), bias = Reactant.ConcretePJRTArray{Float32, 1, 1}(Float32[0.031100241, -0.059878908, 0.08487941, 2.0693283f-6, -0.06550352, -0.08487986, -0.026524307, 0.06386332, 0.042082705, 0.02731392  …  -0.060524136, 0.0346705, -0.02837894, 0.06748107, 0.0027473217, -0.06903143, 0.006823757, 0.014109333, -0.029280972, 0.012123728])), layer_2 = (layer_1 = (weight = Reactant.ConcretePJRTArray{Float32, 2, 1}(Float32[0.120222874 -0.06340912 … -0.0006328318 -0.08487568; 0.05999857 0.08031667 … 0.10135459 -0.008547246; … ; -0.070574954 -0.12529366 … -0.112988465 0.03760839; 0.15816367 -0.12347525 … 0.01897169 0.1261452]), bias = Reactant.ConcretePJRTArray{Float32, 1, 1}(Float32[-0.04772607, -0.0076126982, 0.05891254, -0.05468241, 0.054435007, -0.005947113, -0.03260495, 0.025213236, 0.0024704752, 0.019756647  …  -0.038096465, -0.029946823, 0.021725217, -0.004842341, 0.0018720245, -0.029938111, 0.008292873, 0.01365754, 0.053042553, -0.02033623])), layer_2 = (weight = Reactant.ConcretePJRTArray{Float32, 2, 1}(Float32[-0.10451723 -0.027075663 … 0.018401911 0.06574812; -0.09850936 -0.10326829 … -0.021818167 0.047744956; … ; -0.061319478 -0.06494065 … 0.05045609 0.06725229; -0.025141204 0.094486676 … 0.02057817 -0.09107423]), bias = Reactant.ConcretePJRTArray{Float32, 1, 1}(Float32[-0.03402, 0.055178143, -0.036147784, 0.03546726, -0.026433855, -0.037797134, -0.02685109, -0.05179471, 0.05912261, -0.05172773])))), (layer_1 = NamedTuple(), layer_2 = (layer_1 = NamedTuple(), layer_2 = NamedTuple())), ReactantOptimiser(Adam(eta=Reactant.ConcretePJRTNumber{Float32, 1}(0.0001f0), beta=(Reactant.ConcretePJRTNumber{Float64, 1}(0.9), Reactant.ConcretePJRTNumber{Float64, 1}(0.999)), epsilon=Reactant.ConcretePJRTNumber{Float64, 1}(1.0e-8))), (layer_1 = (weight = [32mLeaf(ReactantOptimiser(Adam(eta=Reactant.ConcretePJRTNumber{Float32, 1}(0.0001f0), beta=(Reactant.ConcretePJRTNumber{Float64, 1}(0.9), Reactant.ConcretePJRTNumber{Float64, 1}(0.999)), epsilon=Reactant.ConcretePJRTNumber{Float64, 1}(1.0e-8))), [39m(Float32[-0.00127828, -0.00129653, -0.000216651, …], Float32[2.05798f-7, 9.33226f-8, 6.09596f-8, …], (Reactant.ConcretePJRTNumber{Float32, 1}(0.729), Reactant.ConcretePJRTNumber{Float32, 1}(0.997003)))[32m)[39m, bias = [32mLeaf(ReactantOptimiser(Adam(eta=Reactant.ConcretePJRTNumber{Float32, 1}(0.0001f0), beta=(Reactant.ConcretePJRTNumber{Float64, 1}(0.9), Reactant.ConcretePJRTNumber{Float64, 1}(0.999)), epsilon=Reactant.ConcretePJRTNumber{Float64, 1}(1.0e-8))), [39m(Float32[-0.00147888, -0.00126067, -0.000245119, …], Float32[2.63239f-7, 8.82631f-8, 6.84694f-8, …], (Reactant.ConcretePJRTNumber{Float32, 1}(0.729), Reactant.ConcretePJRTNumber{Float32, 1}(0.997003)))[32m)[39m), layer_2 = (layer_1 = (weight = [32mLeaf(ReactantOptimiser(Adam(eta=Reactant.ConcretePJRTNumber{Float32, 1}(0.0001f0), beta=(Reactant.ConcretePJRTNumber{Float64, 1}(0.9), Reactant.ConcretePJRTNumber{Float64, 1}(0.999)), epsilon=Reactant.ConcretePJRTNumber{Float64, 1}(1.0e-8))), [39m(Float32[-1.92073f-5, 0.0017717, -0.000850102, …], Float32[3.62355f-11, 1.76661f-7, 4.00854f-8, …], (Reactant.ConcretePJRTNumber{Float32, 1}(0.729), Reactant.ConcretePJRTNumber{Float32, 1}(0.997003)))[32m)[39m, bias = [32mLeaf(ReactantOptimiser(Adam(eta=Reactant.ConcretePJRTNumber{Float32, 1}(0.0001f0), beta=(Reactant.ConcretePJRTNumber{Float64, 1}(0.9), Reactant.ConcretePJRTNumber{Float64, 1}(0.999)), epsilon=Reactant.ConcretePJRTNumber{Float64, 1}(1.0e-8))), [39m(Float32[-7.67666f-6, 0.002996, -0.00215631, …], Float32[7.43951f-12, 4.98398f-7, 2.5891f-7, …], (Reactant.ConcretePJRTNumber{Float32, 1}(0.729), Reactant.ConcretePJRTNumber{Float32, 1}(0.997003)))[32m)[39m), layer_2 = (weight = [32mLeaf(ReactantOptimiser(Adam(eta=Reactant.ConcretePJRTNumber{Float32, 1}(0.0001f0), beta=(Reactant.ConcretePJRTNumber{Float64, 1}(0.9), Reactant.ConcretePJRTNumber{Float64, 1}(0.999)), epsilon=Reactant.ConcretePJRTNumber{Float64, 1}(1.0e-8))), [39m(Float32[-0.00948004, -0.00407593, 0.0052547, …], Float32[5.20286f-6, 4.8477f-6, 2.95658f-6, …], (Reactant.ConcretePJRTNumber{Float32, 1}(0.729), Reactant.ConcretePJRTNumber{Float32, 1}(0.997003)))[32m)[39m, bias = [32mLeaf(ReactantOptimiser(Adam(eta=Reactant.ConcretePJRTNumber{Float32, 1}(0.0001f0), beta=(Reactant.ConcretePJRTNumber{Float64, 1}(0.9), Reactant.ConcretePJRTNumber{Float64, 1}(0.999)), epsilon=Reactant.ConcretePJRTNumber{Float64, 1}(1.0e-8))), [39m(Float32[0.0111157, 0.00553539, -0.0061385, …], Float32[7.24034f-6, 7.03767f-6, 3.97517f-6, …], (Reactant.ConcretePJRTNumber{Float32, 1}(0.729), Reactant.ConcretePJRTNumber{Float32, 1}(0.997003)))[32m)[39m))), 2))
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
Iteration: 0001 	 Loss: 2.08085108
Iteration: 0100 	 Loss: 0.13866809
Iteration: 0200 	 Loss: 0.00399367651
Iteration: 0300 	 Loss: 0.00102405483
Iteration: 0400 	 Loss: 0.000365547283
Iteration: 0500 	 Loss: 0.000224854404
Iteration: 0600 	 Loss: 0.000144167527
Iteration: 0700 	 Loss: 0.000103263519
Iteration: 0800 	 Loss: 0.000436031871
Iteration: 0900 	 Loss: 9.95107548e-05
Iteration: 1000 	 Loss: 0.000275041297
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
