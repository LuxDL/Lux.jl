# # [Training Lux Models using Optimization.jl](@id Optimization-Lux-Tutorial)

# Lux's native [Training.TrainState](@ref) is a great API for gradient-based learning of
# neural networks, however, it is geared towards using `Optimisers.jl` as the backend.
# However, often times we want to train the neural networks with other optimization methods
# like BFGS, LBFGS, etc. In this tutorial, we will show how to train Lux models with
# Optimization.jl that provides a simple unified interface to various optimization methods.

# We will base our tutorial on the minibatching tutorial from the official
# [Optimization.jl](https://docs.sciml.ai/Optimization/stable/tutorials/minibatch/) docs.

# !!! note "Neural ODE"
#
#     This tutorial uses a Neural ODE, however, we won't discuss that part in this tutorial.
#     Please refer to the Neural ODE tutorial for more information.

# ## Imports packages

using Lux, Optimization, OptimizationOptimisers, OptimizationOptimJL, OrdinaryDiffEqTsit5,
      SciMLSensitivity, Random, MLUtils, CairoMakie, ComponentArrays, Printf

const gdev = gpu_device()
const cdev = cpu_device()

# ## Generate some training data

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

# ## Define the DataLoader

# We will define the DataLoader to batch over the data, additionally we will pipe it through
# the `gdev` device to move the data to the GPU on each iteration.

# By default `gdev` will move all objects to the GPU. But we don't want to move the time
# vector to the GPU. So we will wrap it in a struct and mark it as a leaf using
# MLDataDevices.isleaf
struct TimeWrapper{T}
    t::T
end

MLDataDevices.isleaf(::TimeWrapper) = true

Base.length(t::TimeWrapper) = length(t.t)

Base.getindex(t::TimeWrapper, i) = TimeWrapper(t.t[i])

dataloader = DataLoader((ode_data, TimeWrapper(t)); batchsize=8) |> gdev

# ## Training the model

# Here we are using different optimization methods for demonstration purposes. This problem
# is trivial enough to not require this.

# Optimization.jl requires an abstract array as the parameters, hence we will construct a
# `ComponentArray` to store the parameters.

# !!! note "Parameter Estimation vs State Estimation"
#
#     Optimization.jl performs state estimation, which effectively means for a function
#     `f(u, p)`, it is trying to compute the optimal `u` for a given `p`. This terminology
#     might be confusing to ML practitioners, since in the ML world, we usually do parameter
#     estimation. This effectively means that the `u` in Optimization.jl corresponds to our
#     model parameters that is being optimized.

function train_model(dataloader)
    model = Chain(Dense(2, 32, tanh), Dense(32, 32, tanh), Dense(32, 2))
    ps, st = Lux.setup(Random.default_rng(), model)

    ps_ca = ComponentArray(ps) |> gdev
    st = st |> gdev

    function callback(state, l)
        state.iter % 25 == 1 && @printf "Iteration: %5d, Loss: %.6e\n" state.iter l
        return l < 1e-8 ## Terminate if loss is small
    end

    smodel = StatefulLuxLayer{true}(model, nothing, st)

    function loss_adjoint(θ, (u_batch, t_batch))
        t_batch = t_batch.t
        u0 = u_batch[:, 1]
        dudt(u, p, t) = smodel(u, p)
        prob = ODEProblem(dudt, u0, (t_batch[1], t_batch[end]), θ)
        sol = solve(prob, Tsit5(); sensealg=InterpolatingAdjoint(), saveat=t_batch)
        pred = stack(sol.u)
        return MSELoss()(pred, u_batch)
    end

    ## Define the Optimization Function that takes in the optimization state (our parameters)
    ## and optimization parameters (nothing in our case) and data from the dataloader and
    ## returns the loss.
    opt_func = OptimizationFunction(loss_adjoint, Optimization.AutoZygote())
    opt_prob = OptimizationProblem(opt_func, ps_ca, dataloader)

    epochs = 25
    res_adam = solve(opt_prob, Optimisers.Adam(0.001); callback, epochs)

    ## Let's finetune a bit with L-BFGS
    opt_prob = OptimizationProblem(opt_func, res_adam.u, (gdev(ode_data), TimeWrapper(t)))
    res_lbfgs = solve(opt_prob, LBFGS(); callback, maxiters=epochs)

    ## Now that we have a good fit, let's train it on the entire dataset without
    ## Minibatching. We need to do this since ODE solves can lead to accumulated errors if
    ## the model was trained on individual parts (without a data-shooting approach).
    opt_prob = remake(opt_prob; u0=res_lbfgs.u)
    res = solve(opt_prob, Optimisers.Adam(0.005); maxiters=500, callback)

    return StatefulLuxLayer{true}(model, res.u, smodel.st)
end

trained_model = train_model(dataloader)
nothing #hide

# ## Plotting the results

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
