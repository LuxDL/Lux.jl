# [Profiling Lux Training Loops](@id profiling-training-loop-reactant)

!!! warning "Only for Reactant"

    This tutorial is applicable iff you are using `Reactant.jl` (`AutoEnzyme` with
    `ReactantDevice`) for training.

To profile the training loop, wrap the training loop with `Reactant.with_profiler` and
pass the path to the directory where the traces should be saved. Note that this will
have some overhead and hence should be used only for debugging purposes.

A simple example is shown below:

```@example
using Reactant, Lux, Random, MLUtils, Optimisers

dev = reactant_device()

x_data = rand(Float32, 32, 1024)
y_data = x_data .^ 2 .- 1

dl = DataLoader((x_data, y_data); batchsize=32, shuffle=true) |> dev;

model = Chain(Dense(32 => 64, relu), Dense(64 => 32))
ps, st = Lux.setup(Random.default_rng(), model) |> dev;

Reactant.with_profiler(joinpath(tempdir(), "lux_training_trace")) do
    train_state = Training.TrainState(model, ps, st, Adam(0.001))
    for epoch in 1:10
        for (x, y) in dl
            _, loss, _, train_state = Training.single_train_step!(
                AutoEnzyme(), MSELoss(), (x, y), train_state; return_gradients=Val(false)
            )
        end
    end
end
```

Once the run is completed, you can use [`xprof`](https://github.com/openxla/xprof) to
analyze the traces. An example of the output is shown below:

![xprof output](https://github.com/user-attachments/assets/4e27cb09-88d3-40c7-9649-1bde42be4deb)
