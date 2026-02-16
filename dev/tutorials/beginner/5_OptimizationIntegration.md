---
url: /dev/tutorials/beginner/5_OptimizationIntegration.md
---
# Fitting with Optimization.jl {#Optimization-Lux-Tutorial}

Lux's native [Training.TrainState](/api/Lux/utilities#Lux.Training.TrainState) is a great API for gradient-based learning of neural networks, however, it is geared towards using `Optimisers.jl` as the backend. However, often times we want to train the neural networks with other optimization methods like BFGS, LBFGS, etc. In this tutorial, we will show how to train Lux models with Optimization.jl that provides a simple unified interface to various optimization methods.

We will base our tutorial on the minibatching tutorial from the official [Optimization.jl](https://docs.sciml.ai/Optimization/stable/tutorials/minibatch/) docs.

::: tip Neural ODE

This tutorial uses a Neural ODE, however, we won't discuss that part in this tutorial. Please refer to the Neural ODE tutorial for more information.

:::

## Imports packages {#Imports-packages}

```julia
using Lux,
    Optimization,
    OptimizationOptimisers,
    OptimizationOptimJL,
    OrdinaryDiffEqTsit5,
    SciMLSensitivity,
    Random,
    MLUtils,
    CairoMakie,
    ComponentArrays,
    Printf

const gdev = gpu_device()
const cdev = cpu_device()
```

```
┌ Warning: No functional GPU backend found! Defaulting to CPU.
│ 
│ 1. If no GPU is available, nothing needs to be done. Set `MLDATADEVICES_SILENCE_WARN_NO_GPU=1` to silence this warning.
│ 2. If GPU is available, load the corresponding trigger package.
│     a. `CUDA.jl` and `cuDNN.jl` (or just `LuxCUDA.jl`) for  NVIDIA CUDA Support.
│     b. `AMDGPU.jl` for AMD GPU ROCM Support.
│     c. `Metal.jl` for Apple Metal GPU Support. (Experimental)
│     d. `oneAPI.jl` for Intel oneAPI GPU Support. (Experimental)
│     e. `OpenCL.jl` for OpenCL support. (Experimental)
└ @ MLDataDevices.Internal ~/work/Lux.jl/Lux.jl/lib/MLDataDevices/src/internal.jl:114

```

## Generate some training data {#Generate-some-training-data}

```julia
function lotka_volterra(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = α * x - β * x * y
    du[2] = -δ * y + γ * x * y
    return nothing
end

u0 = [1.0f0, 1.0f0]

datasize = 32
tspan = (0.0f0, 2.0f0)

const t = range(tspan[1], tspan[2]; length=datasize)
true_prob = ODEProblem(lotka_volterra, u0, (tspan[1], tspan[2]), [1.5, 1.0, 3.0, 1.0])
const ode_data = Array(solve(true_prob, Tsit5(); saveat=t))

begin
    fig = Figure()
    ax = CairoMakie.Axis(fig[1, 1])
    lines!(ax, t, ode_data[1, :]; label=L"u_1(t)", color=:blue, linestyle=:dot, linewidth=4)
    lines!(ax, t, ode_data[2, :]; label=L"u_2(t)", color=:red, linestyle=:dot, linewidth=4)
    axislegend(ax; position=:lt)
    fig
end
```

## Define the DataLoader {#Define-the-DataLoader}

We will define the DataLoader to batch over the data, additionally we will pipe it through the `gdev` device to move the data to the GPU on each iteration.

By default `gdev` will move all objects to the GPU. But we don't want to move the time vector to the GPU. So we will wrap it in a struct and mark it as a leaf using MLDataDevices.isleaf

```julia
struct TimeWrapper{T}
    t::T
end

MLDataDevices.isleaf(::TimeWrapper) = true

Base.length(t::TimeWrapper) = length(t.t)

Base.getindex(t::TimeWrapper, i) = TimeWrapper(t.t[i])

dataloader = DataLoader((ode_data, TimeWrapper(t)); batchsize=8) |> gdev
```

## Training the model {#Training-the-model}

Here we are using different optimization methods for demonstration purposes. This problem is trivial enough to not require this.

Optimization.jl requires an abstract array as the parameters, hence we will construct a `ComponentArray` to store the parameters.

::: tip Parameter Estimation vs State Estimation

Optimization.jl performs state estimation, which effectively means for a function `f(u, p)`, it is trying to compute the optimal `u` for a given `p`. This terminology might be confusing to ML practitioners, since in the ML world, we usually do parameter estimation. This effectively means that the `u` in Optimization.jl corresponds to our model parameters that is being optimized.

:::

```julia
function train_model(dataloader)
    model = Chain(Dense(2, 32, tanh), Dense(32, 32, tanh), Dense(32, 2))
    ps, st = Lux.setup(Random.default_rng(), model)

    ps_ca = ComponentArray(ps) |> gdev
    st = st |> gdev

    function callback(state, l)
        if state.iter == 1 || state.iter % 25 == 0
            @printf "Iteration: %5d, Loss: %.6e\n" state.iter l
        end
        return l < 1.0e-8 ## Terminate if loss is small
    end

    smodel = StatefulLuxLayer(model, nothing, st)

    function loss_adjoint(θ, (u_batch, t_batch))
        t_batch = t_batch.t
        u0 = u_batch[:, 1]
        dudt(u, p, t) = smodel(u, p)
        prob = ODEProblem(dudt, u0, (t_batch[1], t_batch[end]), θ)
        sol = solve(prob, Tsit5(); sensealg=InterpolatingAdjoint(), saveat=t_batch)
        pred = stack(sol.u)
        return MSELoss()(pred, u_batch)
    end

    # Define the Optimization Function that takes in the optimization state (our parameters)
    # and optimization parameters (nothing in our case) and data from the dataloader and
    # returns the loss.
    opt_func = OptimizationFunction(loss_adjoint, Optimization.AutoZygote())
    opt_prob = OptimizationProblem(opt_func, ps_ca, dataloader)

    epochs = 25
    res_adam = solve(opt_prob, Optimisers.Adam(0.001); callback, epochs)

    # Let's finetune a bit with L-BFGS
    opt_prob = OptimizationProblem(opt_func, res_adam.u, (gdev(ode_data), TimeWrapper(t)))
    res_lbfgs = solve(opt_prob, LBFGS(); callback, maxiters=epochs)

    # Now that we have a good fit, let's train it on the entire dataset without
    # Minibatching. We need to do this since ODE solves can lead to accumulated errors if
    # the model was trained on individual parts (without a data-shooting approach).
    opt_prob = remake(opt_prob; u0=res_lbfgs.u)
    res = solve(opt_prob, Optimisers.Adam(0.005); maxiters=500, callback)

    return StatefulLuxLayer(model, res.u, smodel.st)
end

trained_model = train_model(dataloader)
```

```
Iteration:     1, Loss: 1.135684e-01
Iteration:    25, Loss: 6.273309e-02
Iteration:    50, Loss: 3.308186e-02
Iteration:    75, Loss: 6.881486e-02
Iteration:   100, Loss: 1.979303e-01
Iteration:   100, Loss: 2.614744e-02
Iteration:     1, Loss: 9.293616e-01
Iteration:    25, Loss: 3.081822e-02
Iteration:     1, Loss: 2.947503e-02
Iteration:    25, Loss: 4.385407e-02
Iteration:    50, Loss: 2.771415e-02
Iteration:    75, Loss: 2.579104e-02
Iteration:   100, Loss: 2.479982e-02
Iteration:   125, Loss: 2.412255e-02
Iteration:   150, Loss: 2.339332e-02
Iteration:   175, Loss: 2.245229e-02
Iteration:   200, Loss: 2.111763e-02
Iteration:   225, Loss: 1.947035e-02
Iteration:   250, Loss: 1.784926e-02
Iteration:   275, Loss: 1.734743e-02
Iteration:   300, Loss: 1.607360e-02
Iteration:   325, Loss: 1.542270e-02
Iteration:   350, Loss: 1.442281e-02
Iteration:   375, Loss: 1.182027e-02
Iteration:   400, Loss: 1.230192e-02
Iteration:   425, Loss: 1.658314e-02
Iteration:   450, Loss: 1.035635e-02
Iteration:   475, Loss: 1.505536e-02
Iteration:   500, Loss: 1.169702e-02
Iteration:   500, Loss: 9.250627e-03

```

## Plotting the results {#Plotting-the-results}

```julia
dudt(u, p, t) = trained_model(u, p)
prob = ODEProblem(dudt, gdev(u0), (tspan[1], tspan[2]), trained_model.ps)
sol = solve(prob, Tsit5(); saveat=t)
pred = convert(AbstractArray, sol) |> cdev

begin
    fig = Figure()
    ax = CairoMakie.Axis(fig[1, 1])
    lines!(ax, t, ode_data[1, :]; label=L"u_1(t)", color=:blue, linestyle=:dot, linewidth=4)
    lines!(ax, t, ode_data[2, :]; label=L"u_2(t)", color=:red, linestyle=:dot, linewidth=4)
    lines!(ax, t, pred[1, :]; label=L"\hat{u}_1(t)", color=:blue, linewidth=4)
    lines!(ax, t, pred[2, :]; label=L"\hat{u}_2(t)", color=:red, linewidth=4)
    axislegend(ax; position=:lt)
    fig
end
```

## Appendix {#Appendix}

```julia
using InteractiveUtils
InteractiveUtils.versioninfo()

if @isdefined(MLDataDevices)
    if @isdefined(CUDA) && MLDataDevices.functional(CUDADevice)
        println()
        CUDA.versioninfo()
    end

    if @isdefined(AMDGPU) && MLDataDevices.functional(AMDGPUDevice)
        println()
        AMDGPU.versioninfo()
    end
end

```

```
Julia Version 1.12.5
Commit 5fe89b8ddc1 (2026-02-09 16:05 UTC)
Build Info:
  Official https://julialang.org release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 4 × AMD EPYC 7763 64-Core Processor
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, znver3)
  GC: Built with stock GC
Threads: 4 default, 1 interactive, 4 GC (on 4 virtual cores)
Environment:
  JULIA_DEBUG = Literate
  LD_LIBRARY_PATH = 
  JULIA_NUM_THREADS = 4
  JULIA_CPU_HARD_MEMORY_LIMIT = 100%
  JULIA_PKG_PRECOMPILE_AUTO = 0

```

***

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
