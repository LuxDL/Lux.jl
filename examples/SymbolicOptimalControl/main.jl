# # Solving Optimal Control Problems with Symbolic Universal Differential Equations

# This tutorial is based on [SciMLSensitivity.jl tutorial](https://docs.sciml.ai/SciMLSensitivity/stable/examples/optimal_control/optimal_control/).
# Instead of using a classical NN architecture, here we will combine the NN with a symbolic
# expression from [DynamicExpressions.jl](https://symbolicml.org/DynamicExpressions.jl) (the
# symbolic engine behind [SymbolicRegression.jl](https://astroautomata.com/SymbolicRegression.jl)
# and [PySR](https://github.com/MilesCranmer/PySR/)).

# Here we will solve a classic optimal control problem with a universal differential
# equation. Let

# $$x^{\prime\prime} = u^3(t)$$

# where we want to optimize our controller $u(t)$ such that the following is minimized:

# $$\mathcal{L}(\theta) = \sum_i \left(\|4 - x(t_i)\|_2 + 2\|x\prime(t_i)\|_2 + \|u(t_i)\|_2\right)$$

# where $i$ is measured on $(0, 8)$ at $0.01$ intervals. To do this, we rewrite the ODE in
# first order form:

# $$x^\prime = v$$
# $$v^\prime = u^3(t)$$

# and thus

# $$\mathcal{L}(\theta) = \sum_i \left(\|4 - x(t_i)\|_2 + 2\|v(t_i)\|_2 + \|u(t_i)\|_2\right)$$

# is our loss function on the first order system. We thus choose a neural network form for 
# $u$ and optimize the equation with respect to this loss. Note that we will first reduce
#  control cost (the last term) by 10x in order to bump the network out of a local minimum.
# This looks like:

# ## Package Imports

using Lux, ComponentArrays, OrdinaryDiffEq, Optimization, OptimizationOptimJL,
      OptimizationOptimisers, SciMLSensitivity, Statistics, Printf, Random
using DynamicExpressions, SymbolicRegression, MLJ, SymbolicUtils, Latexify
using CairoMakie

# ## Helper Functions

function plot_dynamics(sol, us, ts)
    fig = Figure()
    ax = CairoMakie.Axis(fig[1, 1]; xlabel=L"t")
    ylims!(ax, (-6, 6))

    lines!(ax, ts, sol[1, :]; label=L"u_1(t)", linewidth=3)
    lines!(ax, ts, sol[2, :]; label=L"u_2(t)", linewidth=3)

    lines!(ax, ts, vec(us); label=L"u(t)", linewidth=3)

    axislegend(ax; position=:rb)

    return fig
end

# ## Training a Neural Network based UDE

# Let's setup the neural network. For the first part, we won't do any symbolic regression.
# We will plain and simple train a neural network to solve the optimal control problem.

rng = Xoshiro(0)
tspan = (0.0, 8.0)

mlp = Chain(Dense(1 => 4, gelu), Dense(4 => 4, gelu), Dense(4 => 1))

function construct_ude(mlp, solver; kwargs...)
    return @compact(; mlp, solver, kwargs...) do x_in, ps
        x, ts, ret_sol = x_in

        function dudt(du, u, p, t)
            u₁, u₂ = u
            du[1] = u₂
            du[2] = mlp([t], p)[1]^3
            return
        end

        prob = ODEProblem{true}(dudt, x, extrema(ts), ps.mlp)

        sol = solve(prob, solver; saveat=ts,
            sensealg=QuadratureAdjoint(; autojacvec=ReverseDiffVJP(true)), kwargs...)

        us = mlp(reshape(ts, 1, :), ps.mlp)
        ret_sol === Val(true) && @return sol, us
        @return Array(sol), us
    end
end

ude = construct_ude(mlp, Vern9(); abstol=1e-10, reltol=1e-10);

# Here we are going to tuse the same configuration for testing, but this is to show that
# we can setup them up with different ode solve configurations

ude_test = construct_ude(mlp, Vern9(); abstol=1e-10, reltol=1e-10);

function train_model_1(ude, rng, ts_)
    ps, st = Lux.setup(rng, ude)
    ps = ComponentArray{Float64}(ps)
    stateful_ude = StatefulLuxLayer(ude, st)

    ts = collect(ts_)

    function loss_adjoint(θ)
        x, us = stateful_ude(([-4.0, 0.0], ts, Val(false)), θ)
        return mean(abs2, 4 .- x[1, :]) + 2 * mean(abs2, x[2, :]) + 0.1 * mean(abs2, us)
    end

    callback = function (state, l)
        state.iter % 50 == 1 && @printf "Iteration: %5d\tLoss: %10g\n" state.iter l
        return false
    end

    optf = OptimizationFunction((x, p) -> loss_adjoint(x), AutoZygote())
    optprob = OptimizationProblem(optf, ps)
    res1 = solve(optprob, Optimisers.Adam(0.001); callback, maxiters=500)

    optprob = OptimizationProblem(optf, res1.u)
    res2 = solve(optprob, LBFGS(); callback, maxiters=100)

    return StatefulLuxLayer{true}(ude, res2.u, st)
end

trained_ude = train_model_1(ude, rng, 0.0:0.01:8.0)
nothing #hide

#-

sol, us = ude_test(([-4.0, 0.0], 0.0:0.01:8.0, Val(true)), trained_ude.ps, trained_ude.st)[1];
plot_dynamics(sol, us, 0.0:0.01:8.0)

# Now that the system is in a better behaved part of parameter space, we return to the
# original loss function to finish the optimization:
function train_model_2(stateful_ude::StatefulLuxLayer, ts_)
    ts = collect(ts_)

    function loss_adjoint(θ)
        x, us = stateful_ude(([-4.0, 0.0], ts, Val(false)), θ)
        return mean(abs2, 4 .- x[1, :]) .+ 2 * mean(abs2, x[2, :]) .+ mean(abs2, us)
    end

    callback = function (state, l)
        state.iter % 10 == 1 && @printf "Iteration: %5d\tLoss: %10g\n" state.iter l
        return false
    end

    optf = OptimizationFunction((x, p) -> loss_adjoint(x), AutoZygote())
    optprob = OptimizationProblem(optf, stateful_ude.ps)
    res2 = solve(optprob, LBFGS(); callback, maxiters=100)

    return StatefulLuxLayer(stateful_ude.model, res2.u, stateful_ude.st)
end

trained_ude = train_model_2(trained_ude, 0.0:0.01:8.0)
nothing #hide

#-

sol, us = ude_test(([-4.0, 0.0], 0.0:0.01:8.0, Val(true)), trained_ude.ps, trained_ude.st)[1];
plot_dynamics(sol, us, 0.0:0.01:8.0)

# ## Symbolic Regression

# Ok so now we have a trained neural network that solves the optimal control problem. But
# can we replace `Dense(4 => 4, gelu)` with a symbolic expression? Let's try!

# ### Data Generation for Symbolic Regression

# First, we need to generate data for the symbolic regression.

ts = reshape(collect(0.0:0.1:8.0), 1, :)

X_train = mlp[1](ts, trained_ude.ps.mlp.layer_1, trained_ude.st.mlp.layer_1)[1]

# This is the training input data. Now we generate the targets

Y_train = mlp[2](X_train, trained_ude.ps.mlp.layer_2, trained_ude.st.mlp.layer_2)[1]

# ### Fitting the Symbolic Expression

# We will follow the example from [SymbolicRegression.jl docs](https://astroautomata.com/SymbolicRegression.jl/dev/examples/)
# to fit the symbolic expression.

srmodel = MultitargetSRRegressor(;
    binary_operators=[+, -, *, /], niterations=100, save_to_file=false);

# One important note here is to transpose the data because that is how MLJ expects the data
# to be structured (this is in contrast to how Lux or SymbolicRegression expects the data)

mach = machine(srmodel, X_train', Y_train')
fit!(mach; verbosity=0)
r = report(mach)
best_eq = [r.equations[1][r.best_idx[1]], r.equations[2][r.best_idx[2]],
    r.equations[3][r.best_idx[3]], r.equations[4][r.best_idx[4]]]

# Let's see the expressions that SymbolicRegression.jl found. In case you were wondering,
# these expressions are not hardcoded, it is live updated from the output of the code above
# using `Latexify.jl` and the integration of `SymbolicUtils.jl` with `DynamicExpressions.jl`.

eqn1 = latexify(string(node_to_symbolic(best_eq[1], srmodel)); fmt=FancyNumberFormatter(5)) #hide
print("__REPLACEME__$(eqn1.s)__REPLACEME__") #hide
nothing #hide

#-

eqn2 = latexify(string(node_to_symbolic(best_eq[2], srmodel)); fmt=FancyNumberFormatter(5)) #hide
print("__REPLACEME__$(eqn2.s)__REPLACEME__") #hide
nothing #hide

#-

eqn1 = latexify(string(node_to_symbolic(best_eq[3], srmodel)); fmt=FancyNumberFormatter(5)) #hide
print("__REPLACEME__$(eqn1.s)__REPLACEME__") #hide
nothing #hide

#-

eqn2 = latexify(string(node_to_symbolic(best_eq[4], srmodel)); fmt=FancyNumberFormatter(5)) #hide
print("__REPLACEME__$(eqn2.s)__REPLACEME__") #hide
nothing #hide

# ## Combining the Neural Network with the Symbolic Expression

# Now that we have the symbolic expression, we can combine it with the neural network to
# solve the optimal control problem. but we do need to perform some finetuning.

hybrid_mlp = Chain(Dense(1 => 4, gelu),
    DynamicExpressionsLayer(OperatorEnum(; binary_operators=[+, -, *, /]), best_eq),
    Dense(4 => 1); disable_optimizations=true)

# There you have it! It is that easy to take the fitted Symbolic Expression and combine it
# with a neural network. Let's see how it performs before fintetuning.

hybrid_ude = construct_ude(hybrid_mlp, Vern9(); abstol=1e-10, reltol=1e-10);

# We want to reuse the trained neural network parameters, so we will copy them over to the
# new model
st = Lux.initialstates(rng, hybrid_ude)
ps = (;
    mlp=(; layer_1=trained_ude.ps.mlp.layer_1,
        layer_2=Lux.initialparameters(rng, hybrid_mlp[2]),
        layer_3=trained_ude.ps.mlp.layer_3))
ps = ComponentArray(ps)

sol, us = hybrid_ude(([-4.0, 0.0], 0.0:0.01:8.0, Val(true)), ps, st)[1];
plot_dynamics(sol, us, 0.0:0.01:8.0)

# Now that does perform well! But we could finetune this model very easily. We will skip
# that part on CI, but you can do it by using the same training code as above.
