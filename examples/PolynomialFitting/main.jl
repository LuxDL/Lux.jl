# # Fitting a Polynomial using MLP

# In this tutorial we will fit a MultiLayer Perceptron (MLP) on data generated from a
# polynomial.

# ## Package Imports
using Lux
import Pkg #hide
Pkg.activate(joinpath(dirname(pathof(Lux)), "..", "examples")) #hide
using NNlib, Optimisers, Random, Statistics, Zygote, CairoMakie, MakiePublication

# ## Dataset

# Generate 128 datapoints from the polynomial $y = x^2 - 2x$.
function generate_data(rng::AbstractRNG)
    x = reshape(collect(range(-2.0f0, 2.0f0, 128)), (1, 128))
    y = evalpoly.(x, ((0, -2, 1),)) .+ randn(rng, (1, 128)) .* 0.1f0
    return (x, y)
end

# Initialize the random number generator and fetch the dataset.
rng = MersenneTwister()
Random.seed!(rng, 12345)

(x, y) = generate_data(rng)

# Let's visualize the dataset
with_theme(theme_web()) do
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="x", ylabel="y")

    l = lines!(ax, x[1, :], x -> evalpoly(x, (0, -2, 1)); linewidth=3)
    s = scatter!(ax, x[1, :], y[1, :]; markersize=8, color=:orange, strokecolor=:black,
                 strokewidth=1)

    axislegend(ax, [l, s], ["True Quadratic Function", "Data Points"])

    return fig
end

# ## Neural Network

# For this problem, you should not be using a neural network. But let's still do that!
model = Chain(Dense(1 => 16, relu), Dense(16 => 1))

# ## Optimizer

# We will use Adam from Optimisers.jl
opt = Adam(0.03f0)

# ## Loss Function

# We will use the `Lux.Training` API so we need to ensure that our loss function takes 4
# inputs -- model, parameters, states and data. The function must return 3 values -- loss,
# updated_state, and any computed statistics.
function loss_function(model, ps, st, data)
    y_pred, st = Lux.apply(model, data[1], ps, st)
    mse_loss = mean(abs2, y_pred .- data[2])
    return mse_loss, st, ()
end

# ## Training

# First we will create a [`Lux.Training.TrainState`](@ref) which is essentially a
# convenience wrapper over parameters, states and optimizer states.

tstate = Lux.Training.TrainState(rng, model, opt; transform_variables=gpu)

# Now we will use Zygote for our AD requirements.

vjp_rule = Lux.Training.ZygoteVJP()

# Finally the training loop.

function main(tstate::Lux.Training.TrainState, vjp::Lux.Training.AbstractVJP, data::Tuple,
              epochs::Int)
    data = data .|> gpu
    for epoch in 1:epochs
        grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp, loss_function,
                                                                    data, tstate)
        @info epoch=epoch loss=loss
        tstate = Lux.Training.apply_gradients(tstate, grads)
    end
    return tstate
end

tstate = main(tstate, vjp_rule, (x, y), 250)
y_pred = cpu(Lux.apply(tstate.model, gpu(x), tstate.parameters, tstate.states)[1])

# Let's plot the results

with_theme(theme_web()) do
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="x", ylabel="y")

    l = lines!(ax, x[1, :], x -> evalpoly(x, (0, -2, 1)); linewidth=3)
    s1 = scatter!(ax, x[1, :], y[1, :]; markersize=8, color=:orange, strokecolor=:black,
                  strokewidth=1)
    s2 = scatter!(ax, x[1, :], y_pred[1, :]; markersize=8, color=:green, strokecolor=:black,
                  strokewidth=1)

    axislegend(ax, [l, s1, s2], ["True Quadratic Function", "Actual Data", "Predictions"])

    return fig
end
