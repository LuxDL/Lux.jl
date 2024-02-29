# # Fitting a Polynomial using MLP

# In this tutorial we will fit a MultiLayer Perceptron (MLP) on data generated from a
# polynomial.

# ## Package Imports
import Pkg #hide
__DIR = @__DIR__ #hide
pkg_io = open(joinpath(__DIR, "pkg.log"), "w") #hide
Pkg.activate(__DIR; io=pkg_io) #hide
Pkg.instantiate(; io=pkg_io) #hide
Pkg.develop(; path=joinpath(__DIR, "..", ".."), io=pkg_io) #hide
Pkg.precompile(; io=pkg_io) #hide
close(pkg_io) #hide
using Lux, LuxAMDGPU, LuxCUDA, Optimisers, Random, Statistics, Zygote
using CairoMakie, MakiePublication

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
    ax = CairoMakie.Axis(fig[1, 1]; xlabel="x", ylabel="y")

    l = lines!(ax, x[1, :], x -> evalpoly(x, (0, -2, 1)); linewidth=3)
    s = scatter!(ax, x[1, :], y[1, :]; markersize=8, color=:orange,
        strokecolor=:black, strokewidth=1)

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

# First we will create a [`Lux.Experimental.TrainState`](@ref) which is essentially a
# convenience wrapper over parameters, states and optimizer states.

tstate = Lux.Training.TrainState(rng, model, opt)

# Now we will use Zygote for our AD requirements.

vjp_rule = Lux.Training.AutoZygote()

# Finally the training loop.

function main(tstate::Lux.Experimental.TrainState, vjp, data, epochs)
    data = data .|> gpu_device()
    for epoch in 1:epochs
        grads, loss, stats, tstate = Lux.Training.compute_gradients(
            vjp, loss_function, data, tstate)
        println("Epoch: $(epoch) || Loss: $(loss)")
        tstate = Lux.Training.apply_gradients(tstate, grads)
    end
    return tstate
end

dev_cpu = cpu_device()
dev_gpu = gpu_device()

tstate = main(tstate, vjp_rule, (x, y), 250)
y_pred = dev_cpu(Lux.apply(tstate.model, dev_gpu(x), tstate.parameters, tstate.states)[1])
nothing #hide

# Let's plot the results

with_theme(theme_web()) do
    fig = Figure()
    ax = CairoMakie.Axis(fig[1, 1]; xlabel="x", ylabel="y")

    l = lines!(ax, x[1, :], x -> evalpoly(x, (0, -2, 1)); linewidth=3)
    s1 = scatter!(ax, x[1, :], y[1, :]; markersize=8, color=:orange,
        strokecolor=:black, strokewidth=1)
    s2 = scatter!(ax, x[1, :], y_pred[1, :]; markersize=8,
        color=:green, strokecolor=:black, strokewidth=1)

    axislegend(ax, [l, s1, s2], ["True Quadratic Function", "Actual Data", "Predictions"])

    return fig
end
