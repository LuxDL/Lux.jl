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
nothing #hide

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

struct TimeWrapper{T}
    t::T
end

MLDataDevices.isleaf(::TimeWrapper) = true

Base.length(t::TimeWrapper) = length(t.t)

Base.getindex(t::TimeWrapper, i) = TimeWrapper(t.t[i])

dataloader = gdev(DataLoader((ode_data, TimeWrapper(t)); batchsize=8))
nothing #hide

function train_model(dataloader)
    model = Chain(Dense(2, 32, tanh), Dense(32, 32, tanh), Dense(32, 2))
    ps, st = Lux.setup(Random.default_rng(), model)

    ps_ca = gdev(ComponentArray(ps))
    st = gdev(st)

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
nothing #hide

dudt(u, p, t) = trained_model(u, p)
prob = ODEProblem(dudt, gdev(u0), (tspan[1], tspan[2]), trained_model.ps)
sol = solve(prob, Tsit5(); saveat=t)
pred = cdev(convert(AbstractArray, sol))

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

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
