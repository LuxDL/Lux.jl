# [Compiling Lux Models using `Reactant.jl`](@id reactant-compilation)

Quoting the Reactant.jl Readme:

> Reactant takes Julia function and compile it into MLIR and run fancy optimizations on top
> of it, including using EnzymeMLIR for automatic differentiation, and create relevant
> executables for CPU/GPU/TPU via XLA. It presently operates as a tracing system. Compiled
> functions will assume the same control flow pattern as was original taken by objects used
> at compile time, and control flow (e.g. if, for) as well as any type instabilities will be
> removed. The benefits of this approach is immediately making all such code available for
> advanced optimization with little developer effort.

```@example compile_lux_model
using Lux, Reactant, Enzyme, Random, Zygote
using Functors, Optimisers, Printf
```

!!! tip "Running on alternate accelerators"

    `Reactant.set_default_backend("gpu")` sets the default backend to CUDA and
    `Reactant.set_default_backend("tpu")` sets the default backend to TPU.

!!! tip "Using the `TrainState` API"

    If you are using the [`Training.TrainState`](@ref) API, skip to the
    [bottom of this page](@ref compile_lux_model_trainstate) to see how to train the model
    without any of this boilerplate.

We start by defining a simple MLP model:

```@example compile_lux_model
model = Chain(
    Dense(2 => 32, gelu),
    Dense(32 => 32, gelu),
    Dense(32 => 2)
)
ps, st = Lux.setup(Random.default_rng(), model)
```

We then create a random input and output data:

```@example compile_lux_model
x = randn(Float32, 2, 32)
y = x .^ 2
nothing # hide
```

We will use [`reactant_device`](@ref) similar to [`gpu_device`](@ref) to move the arrays to
`Reactant`.

```@example compile_lux_model
const xdev = reactant_device()

x_ra = x |> xdev
y_ra = y |> xdev
ps_ra = ps |> xdev
st_ra = st |> xdev
nothing # hide
```

First let's run the model as we would normally:

```@example compile_lux_model
pred_lux, _ = model(x, ps, Lux.testmode(st))
```

To run it using `XLA` we need to compile the model. We can do this using the
`Reactant.@compile` macro. Note that the inputs need to be moved to the device using
[`reactant_device`](@ref) first.

```@example compile_lux_model
model_compiled = @compile model(x_ra, ps_ra, Lux.testmode(st_ra))
```

Now we can test the difference between the results:

```@example compile_lux_model
pred_compiled, _ = model_compiled(x_ra, ps_ra, Lux.testmode(st_ra))

pred_lux .- Array(pred_compiled)
```

The difference is very small as we would expect. Now, let's try to differentiate the
output of the model. We need to use `Enzyme.jl` to do this.

```@example compile_lux_model
function loss_function(model, ps, st, x, y)
    pred, _ = model(x, ps, st)
    return MSELoss()(pred, y)
end
nothing # hide
```

We will use `Zygote.jl` to compute the gradient of the loss function for the vanilla model.

```@example compile_lux_model
loss_function(model, ps, st, x, y)

∂ps_zyg = only(Zygote.gradient(ps -> loss_function(model, ps, st, x, y), ps))
```

Now we will compile the gradient function using `Reactant.@compile`.

```@example compile_lux_model
function enzyme_gradient(model, ps, st, x, y)
    return Enzyme.gradient(Enzyme.Reverse, Const(loss_function), Const(model),
        ps, Const(st), Const(x), Const(y))[2]
end

enzyme_gradient_compiled = @compile enzyme_gradient(model, ps_ra, st_ra, x_ra, y_ra)

∂ps_enzyme = enzyme_gradient_compiled(model, ps_ra, st_ra, x_ra, y_ra)
```

Now we check the difference:

```@example compile_lux_model
fmap(Broadcast.BroadcastFunction(-), ∂ps_zyg, ∂ps_enzyme |> cpu_device())
```

## [Using the `TrainState` API](@id compile_lux_model_trainstate)

Now that we saw the low-level API let's see how to train the model without any of this
boilerplate. Simply follow the following steps:

1. Create a device using `reactant_device`. Remember to load `Reactant.jl` before doing this.
2. Similar to other device functions move the model, parameters, states and data to the
   device. Note that you might want to use [`DeviceIterator`](@ref) to move the data
   loader to the device with an iterator.
3. Construct a `TrainState` using [`Training.TrainState`](@ref).
4. And most importantly use `AutoEnzyme` while calling [`Training.single_train_step!`](@ref)
   or [`Training.single_train_step`](@ref).

```@example compile_lux_model
model = Chain(
    Dense(2 => 4, gelu),
    Dense(4 => 4, gelu),
    Dense(4 => 2)
)
ps, st = Lux.setup(Random.default_rng(), model)

x_ra = [randn(Float32, 2, 32) for _ in 1:32]
y_ra = [xᵢ .^ 2 for xᵢ in x_ra]
ps_ra = ps |> xdev
st_ra = st |> xdev

dataloader = DeviceIterator(xdev, zip(x_ra, y_ra))

function train_model(model, ps, st, dataloader)
    train_state = Training.TrainState(model, ps, st, Adam(0.001f0))

    for iteration in 1:1000
        for (i, (xᵢ, yᵢ)) in enumerate(dataloader)
            _, loss, _, train_state = Training.single_train_step!(
                AutoEnzyme(), MSELoss(), (xᵢ, yᵢ), train_state)
            if (iteration % 100 == 0 || iteration == 1) && i == 1
                @printf("Iter: [%4d/%4d]\tLoss: %.8f\n", iteration, 1000, loss)
            end
        end
    end

    return train_state
end

train_model(model, ps_ra, st_ra, dataloader)
nothing # hide
```
