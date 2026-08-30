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
I0000 00:00:1788059620.600333    8310 profiler_session.cc:171] Profiler session initializing.
I0000 00:00:1788059620.600370    8310 profiler_session.cc:186] Profiler session started.
I0000 00:00:1788059655.869582    8310 profiler_session.cc:134] Profiler session collecting data.
I0000 00:00:1788059655.954825    8310 save_profile.cc:205] Collecting XSpace to repository: /tmp/lux_training_trace/plugins/profile/2026_08_30_03_14_15/runnervmgx7h7.xplane.pb
I0000 00:00:1788059656.035832    8310 save_profile.cc:178] Creating directory: /tmp/lux_training_trace/plugins/profile/2026_08_30_03_14_15

I0000 00:00:1788059656.102705    8310 save_profile.cc:184] Dumped gzipped tool data for trace.json.gz to /tmp/lux_training_trace/plugins/profile/2026_08_30_03_14_15/runnervmgx7h7.trace.json.gz
I0000 00:00:1788059656.136434    8310 profiler_session.cc:217] Profiler session tear down.
```

Once the run is completed, you can use [`xprof`](https://github.com/openxla/xprof) to analyze the traces. An example of the output is shown below:
