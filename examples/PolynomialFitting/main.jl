# # Fitting a Polynomial using MLP

# In this tutorial we will fit a MultiLayer Perceptron (MLP) on data generated from a
# polynomial.

# ## Package Imports

using Lux, ADTypes, Optimisers, Printf, Random, Reactant, Statistics, CairoMakie

# ## Dataset

# Generate 128 datapoints from the polynomial $y = x^2 - 2x$.
function generate_data(rng::AbstractRNG)
    x = reshape(collect(range(-2.0f0, 2.0f0, 128)), (1, 128))
    y = evalpoly.(x, ((0, -2, 1),)) .+ randn(rng, Float32, (1, 128)) .* 0.1f0
    return (x, y)
end

# Initialize the random number generator and fetch the dataset.
rng = MersenneTwister()
Random.seed!(rng, 12345)

(x, y) = generate_data(rng)

# Let's visualize the dataset
begin
    fig = Figure()
    ax = CairoMakie.Axis(fig[1, 1]; xlabel="x", ylabel="y")

    l = lines!(ax, x[1, :], x -> evalpoly(x, (0, -2, 1)); linewidth=3, color=:blue)
    s = scatter!(ax, x[1, :], y[1, :]; markersize=12, alpha=0.5,
        color=:orange, strokecolor=:black, strokewidth=2)

    axislegend(ax, [l, s], ["True Quadratic Function", "Data Points"])

    fig
end

# ## Neural Network

# For this problem, you should not be using a neural network. But let's still do that!
model = Chain(Dense(1 => 16, relu), Dense(16 => 1))

# ## Optimizer

# We will use Adam from Optimisers.jl
opt = Adam(0.03f0)

# ## Loss Function

# We will use the `Training` API so we need to ensure that our loss function takes 4
# inputs -- model, parameters, states and data. The function must return 3 values -- loss,
# updated_state, and any computed statistics. This is already satisfied by the loss
# functions provided by Lux.
const loss_function = MSELoss()

const cdev = cpu_device()
const xdev = reactant_device()

ps, st = Lux.setup(rng, model) |> xdev

# ## Training

# First we will create a [`Training.TrainState`](@ref) which is essentially a
# convenience wrapper over parameters, states and optimizer states.

tstate = Training.TrainState(model, ps, st, opt)

# Now we will use Enzyme (Reactant) for our AD requirements.

vjp_rule = AutoEnzyme()

# Finally the training loop.

function main(tstate::Training.TrainState, vjp, data, epochs)
    data = data |> xdev
    for epoch in 1:epochs
        _, loss, _, tstate = Training.single_train_step!(vjp, loss_function, data, tstate)
        if epoch % 50 == 1 || epoch == epochs
            @printf "Epoch: %3d \t Loss: %.5g\n" epoch loss
        end
    end
    return tstate
end

tstate = main(tstate, vjp_rule, (x, y), 250)

# Since we are using Reactant, we need to compile the model before we can use it.

forward_pass = @compile Lux.apply(
    tstate.model, xdev(x), tstate.parameters, Lux.testmode(tstate.states)
)

y_pred = cdev(first(forward_pass(
    tstate.model, xdev(x), tstate.parameters, Lux.testmode(tstate.states)
)))
nothing #hide

# Let's plot the results

begin
    fig = Figure()
    ax = CairoMakie.Axis(fig[1, 1]; xlabel="x", ylabel="y")

    l = lines!(ax, x[1, :], x -> evalpoly(x, (0, -2, 1)); linewidth=3)
    s1 = scatter!(ax, x[1, :], y[1, :]; markersize=12, alpha=0.5,
        color=:orange, strokecolor=:black, strokewidth=2)
    s2 = scatter!(ax, x[1, :], y_pred[1, :]; markersize=12, alpha=0.5,
        color=:green, strokecolor=:black, strokewidth=2)

    axislegend(ax, [l, s1, s2], ["True Quadratic Function", "Actual Data", "Predictions"])

    fig
end
