# # Fitting a Polynomial using MLP

# In this tutorial we will fit a MultiLayer Perceptron (MLP) on data generated from a
# polynomial.

# ## Package Imports
import Lux
import Pkg #hide
Pkg.activate(joinpath(dirname(pathof(Lux)), "..", "examples")) #hide
import NNlib, Optimisers, Plots, Random, Statistics, Zygote

# ## Dataset

# Generate 128 datapoints from the polynomial $y = x^2 - 2x$.
function generate_data(rng::Random.AbstractRNG)
    x = reshape(collect(range(-2.0f0, 2.0f0, 128)), (1, 128))
    y = evalpoly.(x, ((0, -2, 1),)) .+ randn(rng, (1, 128)) .* 0.1f0
    return (x, y)
end

# Initialize the random number generator and fetch the dataset.
rng = Random.MersenneTwister()
Random.seed!(rng, 12345)

(x, y) = generate_data(rng)

# Let's visualize the dataset
Plots.plot(x -> evalpoly(x, (0, -2, 1)), x[1, :]; label=false)
Plots.scatter!(x[1, :], y[1, :]; label=false, markersize=3)

# ## Neural Network

# For this problem, you should not be using a neural network. But let's still do that!
function construct_model()
    return Lux.Chain(Lux.Dense(1, 16, NNlib.relu), Lux.Dense(16, 1))
end

model = construct_model()

# ## Optimizer

# We will use Adam from Optimisers.jl
opt = Optimisers.Adam(0.03)

# ## Loss Function

# We will use the `Lux.Training` API so we need to ensure that our loss function takes 4
# inputs -- model, parameters, states and data. The function must return 3 values -- loss,
# updated_state, and any computed statistics.
function loss_function(model, ps, st, data)
    y_pred, st = Lux.apply(model, data[1], ps, st)
    mse_loss = Statistics.mean(abs2, y_pred .- data[2])
    return mse_loss, st, ()
end

# ## Training

# First we will create a [`Lux.Training.TrainState`](@ref) which is essentially a
# convenience wrapper over parameters, states and optimizer states.

tstate = Lux.Training.TrainState(rng, model, opt; transform_variables=Lux.gpu)

# Now we will use Zygote for our AD requirements.

vjp_rule = Lux.Training.ZygoteVJP()

# Finally the training loop.

function main(tstate::Lux.Training.TrainState, vjp::Lux.Training.AbstractVJP, data::Tuple,
              epochs::Int)
    data = data .|> Lux.gpu
    for epoch in 1:epochs
        grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp, loss_function,
                                                                    data, tstate)
        @info epoch=epoch loss=loss
        tstate = Lux.Training.apply_gradients(tstate, grads)
    end
    return tstate
end

tstate = main(tstate, vjp_rule, (x, y), 250)
y_pred = Lux.cpu(Lux.apply(tstate.model, Lux.gpu(x), tstate.parameters, tstate.states)[1])

# Let's plot the results

Plots.plot(x -> evalpoly(x, (0, -2, 1)), x[1, :]; label=false)
Plots.scatter!(x[1, :], y[1, :]; label="Actual Data", markersize=3)
Plots.scatter!(x[1, :], y_pred[1, :]; label="Predictions", markersize=3)
