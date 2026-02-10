---
url: /dev/manual/profiling_training_loop.md
---
# Profiling Lux Training Loops {#profiling-training-loop-reactant}

::: warning Only for Reactant

This tutorial is applicable iff you are using `Reactant.jl` (`AutoEnzyme` with `ReactantDevice`) for training.

:::

To profile the training loop, wrap the training loop with `Reactant.with_profiler` and pass the path to the directory where the traces should be saved. Note that this will have some overhead and hence should be used only for debugging purposes.

A simple example is shown below:

```julia
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

```ansi
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1770735803.260176   11530 profiler_session.cc:117] Profiler session initializing.
I0000 00:00:1770735803.260207   11530 profiler_session.cc:132] Profiler session started.
I0000 00:00:1770735834.865212   11530 profiler_session.cc:81] Profiler session collecting data.
I0000 00:00:1770735834.933187   11530 save_profile.cc:150] Collecting XSpace to repository: /tmp/lux_training_trace/plugins/profile/2026_02_10_15_03_54/runnervmwffz4.xplane.pb
I0000 00:00:1770735834.991309   11530 save_profile.cc:123] Creating directory: /tmp/lux_training_trace/plugins/profile/2026_02_10_15_03_54

I0000 00:00:1770735835.041839   11530 save_profile.cc:129] Dumped gzipped tool data for trace.json.gz to /tmp/lux_training_trace/plugins/profile/2026_02_10_15_03_54/runnervmwffz4.trace.json.gz
I0000 00:00:1770735835.062200   11530 profiler_session.cc:150] Profiler session tear down.
```

Once the run is completed, you can use [`xprof`](https://github.com/openxla/xprof) to analyze the traces. An example of the output is shown below:
