# # Bayesian Neural Network

# We borrow this tutorial from the
# [official Turing Docs](https://turinglang.org/docs/tutorials/03-bayesian-neural-network/index.html).
# We will show how the explicit parameterization of Lux enables first-class composability
# with packages which expect flattened out parameter vectors.

# Note: The tutorial in the official Turing docs is now using Lux instead of Flux.

# We will use [Turing.jl](https://turinglang.org/) with [Lux.jl](https://lux.csail.mit.edu/)
# to implement implementing a classification algorithm. Lets start by importing the relevant
# libraries.

## Import libraries

using Lux, Turing, CairoMakie, Random, Tracker, Functors, LinearAlgebra

## Sampling progress
Turing.setprogress!(true);

# ## Generating data

# Our goal here is to use a Bayesian neural network to classify points in an artificial
# dataset. The code below generates data points arranged in a box-like pattern and displays
# a graph of the dataset we'll be working with.

## Number of points to generate
N = 80
M = round(Int, N / 4)
rng = Random.default_rng()
Random.seed!(rng, 1234)

## Generate artificial data
x1s = rand(rng, Float32, M) * 4.5f0;
x2s = rand(rng, Float32, M) * 4.5f0;
xt1s = Array([[x1s[i] + 0.5f0; x2s[i] + 0.5f0] for i in 1:M])
x1s = rand(rng, Float32, M) * 4.5f0;
x2s = rand(rng, Float32, M) * 4.5f0;
append!(xt1s, Array([[x1s[i] - 5.0f0; x2s[i] - 5.0f0] for i in 1:M]))

x1s = rand(rng, Float32, M) * 4.5f0;
x2s = rand(rng, Float32, M) * 4.5f0;
xt0s = Array([[x1s[i] + 0.5f0; x2s[i] - 5.0f0] for i in 1:M])
x1s = rand(rng, Float32, M) * 4.5f0;
x2s = rand(rng, Float32, M) * 4.5f0;
append!(xt0s, Array([[x1s[i] - 5.0f0; x2s[i] + 0.5f0] for i in 1:M]))

## Store all the data for later
xs = [xt1s; xt0s]
ts = [ones(2 * M); zeros(2 * M)]

## Plot data points

function plot_data()
    x1 = first.(xt1s)
    y1 = last.(xt1s)
    x2 = first.(xt0s)
    y2 = last.(xt0s)

    fig = Figure()
    ax = CairoMakie.Axis(fig[1, 1]; xlabel="x", ylabel="y")

    scatter!(ax, x1, y1; markersize=16, color=:red, strokecolor=:black, strokewidth=2)
    scatter!(ax, x2, y2; markersize=16, color=:blue, strokecolor=:black, strokewidth=2)

    return fig
end

plot_data()

# ## Building the Neural Network

# The next step is to define a feedforward neural network where we express our parameters as
# distributions, and not single points as with traditional neural networks. For this we will
# use `Dense` to define liner layers and compose them via `Chain`, both are neural network
# primitives from `Lux`. The network `nn` we will create will have two hidden layers with
# `tanh` activations and one output layer with `sigmoid` activation, as shown below.

# The `nn` is an instance that acts as a function and can take data, parameters and current
# state as inputs and output predictions. We will define distributions on the neural network
# parameters.

## Construct a neural network using Lux
nn = Chain(Dense(2 => 3, tanh), Dense(3 => 2, tanh), Dense(2 => 1, sigmoid))

## Initialize the model weights and state
ps, st = Lux.setup(rng, nn)

Lux.parameterlength(nn) # number of parameters in NN

# The probabilistic model specification below creates a parameters variable, which has IID
# normal variables. The parameters represents all parameters of our neural net (weights and
# biases).

## Create a regularization term and a Gaussian prior variance term.
alpha = 0.09
sig = sqrt(1.0 / alpha)

# Construct named tuple from a sampled parameter vector. We could also use ComponentArrays
# here and simply broadcast to avoid doing this. But let's do it this way to avoid
# dependencies.
function vector_to_parameters(ps_new::AbstractVector, ps::NamedTuple)
    @assert length(ps_new) == Lux.parameterlength(ps)
    i = 1
    function get_ps(x)
        z = reshape(view(ps_new, i:(i + length(x) - 1)), size(x))
        i += length(x)
        return z
    end
    return fmap(get_ps, ps)
end

# To interface with external libraries it is often desirable to use the
# [`StatefulLuxLayer`](@ref) to automatically handle the neural network states.
const model = StatefulLuxLayer{true}(nn, nothing, st)

## Specify the probabilistic model.
@model function bayes_nn(xs, ts)
    ## Sample the parameters
    nparameters = Lux.parameterlength(nn)
    parameters ~ MvNormal(zeros(nparameters), Diagonal(abs2.(sig .* ones(nparameters))))

    ## Forward NN to make predictions
    preds = Lux.apply(model, xs, vector_to_parameters(parameters, ps))

    ## Observe each prediction.
    for i in eachindex(ts)
        ts[i] ~ Bernoulli(preds[i])
    end
end

# Inference can now be performed by calling sample. We use the HMC sampler here.

## Perform inference.
N = 5000
ch = sample(bayes_nn(reduce(hcat, xs), ts), HMC(0.05, 4; adtype=AutoTracker()), N)

# Now we extract the parameter samples from the sampled chain as θ (this is of size
# `5000 x 20` where `5000` is the number of iterations and `20` is the number of
# parameters). We'll use these primarily to determine how good our model's classifier is.

## Extract all weight and bias parameters.
θ = MCMCChains.group(ch, :parameters).value;

# ## Prediction Visualization

## A helper to run the nn through data `x` using parameters `θ`
nn_forward(x, θ) = model(x, vector_to_parameters(θ, ps))

## Plot the data we have.
fig = plot_data()

## Find the index that provided the highest log posterior in the chain.
_, i = findmax(ch[:lp])

## Extract the max row value from i.
i = i.I[1]

## Plot the posterior distribution with a contour plot
x1_range = collect(range(-6; stop=6, length=25))
x2_range = collect(range(-6; stop=6, length=25))
Z = [nn_forward([x1, x2], θ[i, :])[1] for x1 in x1_range, x2 in x2_range]
contour!(x1_range, x2_range, Z; linewidth=3, colormap=:seaborn_bright)
fig

# The contour plot above shows that the MAP method is not too bad at classifying our data.
# Now we can visualize our predictions.

# $p(\tilde{x} | X, \alpha) = \int_{\theta} p(\tilde{x} | \theta) p(\theta | X, \alpha) \approx \sum_{\theta \sim p(\theta | X, \alpha)}f_{\theta}(\tilde{x})$

# The `nn_predict` function takes the average predicted value from a network parameterized
# by weights drawn from the MCMC chain.

## Return the average predicted value across multiple weights.
nn_predict(x, θ, num) = mean([first(nn_forward(x, view(θ, i, :))) for i in 1:10:num])

# Next, we use the `nn_predict` function to predict the value at a sample of points where
# the x1 and x2 coordinates range between -6 and 6. As we can see below, we still have a
# satisfactory fit to our data, and more importantly, we can also see where the neural
# network is uncertain about its predictions much easier---those regions between cluster
# boundaries.

# Plot the average prediction.
fig = plot_data()

n_end = 1500
x1_range = collect(range(-6; stop=6, length=25))
x2_range = collect(range(-6; stop=6, length=25))
Z = [nn_predict([x1, x2], θ, n_end)[1] for x1 in x1_range, x2 in x2_range]
contour!(x1_range, x2_range, Z; linewidth=3, colormap=:seaborn_bright)
fig

# Suppose we are interested in how the predictive power of our Bayesian neural network
# evolved between samples. In that case, the following graph displays an animation of the
# contour plot generated from the network weights in samples 1 to 5,000.

fig = plot_data()
Z = [first(nn_forward([x1, x2], θ[1, :])) for x1 in x1_range, x2 in x2_range]
c = contour!(x1_range, x2_range, Z; linewidth=3, colormap=:seaborn_bright)
record(fig, "results.gif", 1:250:size(θ, 1)) do i
    fig.current_axis[].title = "Iteration: $i"
    Z = [first(nn_forward([x1, x2], θ[i, :])) for x1 in x1_range, x2 in x2_range]
    c[3] = Z
    return fig
end

# ![](results.gif)
