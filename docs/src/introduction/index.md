# [Introduction](@id getting-started)

## Installation

Install [Julia v1.10 or above](https://julialang.org/downloads/). Lux.jl is available
through the Julia package manager. You can enter it by pressing `]` in the REPL and then
typing `add Lux`. Alternatively, you can also do

```julia
import Pkg
Pkg.add("Lux")
```

!!! tip "Update to v1"

    If you are using a pre-v1 version of Lux.jl, please see the
    [Updating to v1 section](@ref updating-to-v1) for instructions on how to update.

## Quickstart

!!! tip "Pre-Requisites"

    You need to install `Optimisers`, `Reactant` and `Enzyme` if not done already.
    `Pkg.add(["Optimisers", "Enzyme", "Reactant"])`

```@example quickstart
using Lux, Random, Optimisers, Enzyme, Reactant
```

We take randomness very seriously

```@example quickstart
# Seeding
rng = Random.default_rng()
Random.seed!(rng, 0)
```

Build the model

```@example quickstart
# Construct the layer
model = Chain(Dense(128, 256, tanh), Chain(Dense(256, 256, tanh), Dense(256, 10)))
```

Models don't hold parameters and states so initialize them. From there on, we can just use
our standard AD and Optimisers API. However, here we will show how to use Lux's Training
API that provides an uniform API over all supported AD systems.

```@example quickstart
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

## Defining Custom Layers

We can train our model using the above code, but let's go ahead and see how to use Reactant.
Reactant is a julia frontend that generates MLIR and then compiles it using XLA (after
running fancy optimizations). It is the current recommended way to train large models in
Lux. For more details on using Reactant, see the [manual](@ref reactant-compilation).

```@example custom_compact
using Lux, Random, Optimisers, Reactant, Enzyme
using Printf # For pretty printing

dev = reactant_device()
```

We will define a custom MLP using the `@compact` macro. The macro takes in a list of
parameters, layers and states, and a function defining the forward pass of the neural
network.

```@example custom_compact
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

We can initialize the model and train it with the same code as before!

```@example custom_compact
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
nothing #hide
```

!!! tip "Training with Optimization.jl"

    If you are coming from the SciML ecosystem and want to use Optimization.jl, please
    refer to the [Optimization.jl Tutorial](@ref Optimization-Lux-Tutorial).

## Additional Packages

`LuxDL` hosts various packages that provide additional functionality for Lux.jl. All
packages mentioned in this documentation are available via the Julia General Registry.

You can install all those packages via `import Pkg; Pkg.add(<package name>)`.

## Reactant & XLA (CPU/GPU/TPU) Support

Lux.jl supports XLA compilation for CPU, GPU, and TPU using
[Reactant.jl](https://github.com/EnzymeAD/Reactant.jl).

## Native Julia GPU Support

!!! warning

    Using accelerators via Reactant and XLA is the preferred way to use GPUs with Lux.jl.

!!! danger "AMD GPU Users"

    For AMD GPUs, we **strongly recommend using Reactant** instead of native AMDGPU.jl
    support. Native AMDGPU.jl support is experimental with known issues including deadlocks.
    Reactant provides superior performance and reliability for AMD hardware.

GPU Support for Lux.jl requires loading additional packages:

- [`LuxCUDA.jl`](https://github.com/LuxDL/LuxCUDA.jl) for CUDA support.
- [`AMDGPU.jl`](https://github.com/JuliaGPU/AMDGPU.jl) for AMDGPU support (not recommended, use Reactant instead).
- [`Metal.jl`](https://github.com/JuliaGPU/Metal.jl) for Apple Metal support.
- [`oneAPI.jl`](https://github.com/JuliaGPU/oneAPI.jl) for oneAPI support.
