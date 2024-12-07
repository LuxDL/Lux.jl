# # MNIST Classification using Neural ODEs

# To understand Neural ODEs, users should look up
# [these lecture notes](https://book.sciml.ai/notes/11-Differentiable_Programming_and_Neural_Differential_Equations/).
# We recommend users to directly use
# [DiffEqFlux.jl](https://docs.sciml.ai/DiffEqFlux/stable/), instead of implementing
# Neural ODEs from scratch.

# ## Package Imports

using Lux, ComponentArrays, SciMLSensitivity, LuxCUDA, Optimisers, OrdinaryDiffEqTsit5,
      Random, Statistics, Zygote, OneHotArrays, InteractiveUtils, Printf
using MLDatasets: MNIST
using MLUtils: DataLoader, splitobs

CUDA.allowscalar(false)

# ## Loading MNIST
function loadmnist(batchsize, train_split)
    ## Load MNIST: Only 1500 for demonstration purposes
    N = parse(Bool, get(ENV, "CI", "false")) ? 1500 : nothing
    dataset = MNIST(; split=:train)
    if N !== nothing
        imgs = dataset.features[:, :, 1:N]
        labels_raw = dataset.targets[1:N]
    else
        imgs = dataset.features
        labels_raw = dataset.targets
    end

    ## Process images into (H,W,C,BS) batches
    x_data = Float32.(reshape(imgs, size(imgs, 1), size(imgs, 2), 1, size(imgs, 3)))
    y_data = onehotbatch(labels_raw, 0:9)
    (x_train, y_train), (x_test, y_test) = splitobs((x_data, y_data); at=train_split)

    return (
        ## Use DataLoader to automatically minibatch and shuffle the data
        DataLoader(collect.((x_train, y_train)); batchsize, shuffle=true),
        ## Don't shuffle the test data
        DataLoader(collect.((x_test, y_test)); batchsize, shuffle=false)
    )
end

# ## Define the Neural ODE Layer
#
# First we will use the [`@compact`](@ref) macro to define the Neural ODE Layer.

function NeuralODECompact(
        model::Lux.AbstractLuxLayer; solver=Tsit5(), tspan=(0.0f0, 1.0f0), kwargs...)
    return @compact(; model, solver, tspan, kwargs...) do x, p
        dudt(u, p, t) = vec(model(reshape(u, size(x)), p))
        ## Note the `p.model` here
        prob = ODEProblem(ODEFunction{false}(dudt), vec(x), tspan, p.model)
        @return solve(prob, solver; kwargs...)
    end
end

# We recommend using the compact macro for creating custom layers. The below implementation
# exists mostly for historical reasons when `@compact` was not part of the stable API. Also,
# it helps users understand how the layer interface of Lux works.

# The NeuralODE is a ContainerLayer, which stores a `model`. The parameters and states of
# the NeuralODE are same as those of the underlying model.
struct NeuralODE{M <: Lux.AbstractLuxLayer, So, T, K} <: Lux.AbstractLuxWrapperLayer{:model}
    model::M
    solver::So
    tspan::T
    kwargs::K
end

function NeuralODE(
        model::Lux.AbstractLuxLayer; solver=Tsit5(), tspan=(0.0f0, 1.0f0), kwargs...)
    return NeuralODE(model, solver, tspan, kwargs)
end

# OrdinaryDiffEq.jl can deal with non-Vector Inputs! However, certain discrete sensitivities
# like `ReverseDiffAdjoint` can't handle non-Vector inputs. Hence, we need to convert the
# input and output of the ODE solver to a Vector.
function (n::NeuralODE)(x, ps, st)
    function dudt(u, p, t)
        u_, st = n.model(reshape(u, size(x)), p, st)
        return vec(u_)
    end
    prob = ODEProblem{false}(ODEFunction{false}(dudt), vec(x), n.tspan, ps)
    return solve(prob, n.solver; n.kwargs...), st
end

@views diffeqsol_to_array(l::Int, x::ODESolution) = reshape(last(x.u), (l, :))
@views diffeqsol_to_array(l::Int, x::AbstractMatrix) = reshape(x[:, end], (l, :))

# ## Create and Initialize the Neural ODE Layer
function create_model(model_fn=NeuralODE; dev=gpu_device(), use_named_tuple::Bool=false,
        sensealg=InterpolatingAdjoint(; autojacvec=ZygoteVJP()))
    ## Construct the Neural ODE Model
    model = Chain(FlattenLayer(),
        Dense(784 => 20, tanh),
        model_fn(
            Chain(Dense(20 => 10, tanh), Dense(10 => 10, tanh), Dense(10 => 20, tanh));
            save_everystep=false, reltol=1.0f-3,
            abstol=1.0f-3, save_start=false, sensealg),
        Base.Fix1(diffeqsol_to_array, 20),
        Dense(20 => 10))

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    ps, st = Lux.setup(rng, model)
    ps = (use_named_tuple ? ps : ComponentArray(ps)) |> dev
    st = st |> dev

    return model, ps, st
end

# ## Define Utility Functions
const logitcrossentropy = CrossEntropyLoss(; logits=Val(true))

function accuracy(model, ps, st, dataloader)
    total_correct, total = 0, 0
    st = Lux.testmode(st)
    for (x, y) in dataloader
        target_class = onecold(y)
        predicted_class = onecold(first(model(x, ps, st)))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end

# ## Training
function train(model_function; cpu::Bool=false, kwargs...)
    dev = cpu ? cpu_device() : gpu_device()
    model, ps, st = create_model(model_function; dev, kwargs...)

    ## Training
    train_dataloader, test_dataloader = loadmnist(128, 0.9) |> dev

    tstate = Training.TrainState(model, ps, st, Adam(0.001f0))

    ### Lets train the model
    nepochs = 9
    for epoch in 1:nepochs
        stime = time()
        for (x, y) in train_dataloader
            _, _, _, tstate = Training.single_train_step!(
                AutoZygote(), logitcrossentropy, (x, y), tstate)
        end
        ttime = time() - stime

        tr_acc = accuracy(model, tstate.parameters, tstate.states, train_dataloader) * 100
        te_acc = accuracy(model, tstate.parameters, tstate.states, test_dataloader) * 100
        @printf "[%d/%d]\tTime %.4fs\tTraining Accuracy: %.5f%%\tTest \
                 Accuracy: %.5f%%\n" epoch nepochs ttime tr_acc te_acc
    end
end

train(NeuralODECompact)
nothing #hide

#-

train(NeuralODE)
nothing #hide

# We can also change the sensealg and train the model! `GaussAdjoint` allows you to use
# any arbitrary parameter structure and not just a flat vector (`ComponentArray`).

train(NeuralODE; sensealg=GaussAdjoint(; autojacvec=ZygoteVJP()), use_named_tuple=true)

# But remember some AD backends like `ReverseDiff` is not GPU compatible.
# For a model this size, you will notice that training time is significantly lower for
# training on CPU than on GPU.

train(NeuralODE; sensealg=InterpolatingAdjoint(; autojacvec=ReverseDiffVJP()), cpu=true)

# For completeness, let's also test out discrete sensitivities!

train(NeuralODE; sensealg=ReverseDiffAdjoint(), cpu=true)

# ## Alternate Implementation using Stateful Layer

# Starting `v0.5.5`, Lux provides a [`StatefulLuxLayer`](@ref) which can be used
# to avoid the [`Box`ing of `st`](https://github.com/JuliaLang/julia/issues/15276). Using
# the `@compact` API avoids this problem entirely.
struct StatefulNeuralODE{M <: Lux.AbstractLuxLayer, So, T, K} <:
       Lux.AbstractLuxWrapperLayer{:model}
    model::M
    solver::So
    tspan::T
    kwargs::K
end

function StatefulNeuralODE(
        model::Lux.AbstractLuxLayer; solver=Tsit5(), tspan=(0.0f0, 1.0f0), kwargs...)
    return StatefulNeuralODE(model, solver, tspan, kwargs)
end

function (n::StatefulNeuralODE)(x, ps, st)
    st_model = StatefulLuxLayer{true}(n.model, ps, st)
    dudt(u, p, t) = st_model(u, p)
    prob = ODEProblem{false}(ODEFunction{false}(dudt), x, n.tspan, ps)
    return solve(prob, n.solver; n.kwargs...), st_model.st
end

# ## Train the new Stateful Neural ODE
train(StatefulNeuralODE)

# We might not see a significant difference in the training time, but let us investigate
# the type stabilities of the layers.

# ## Type Stability

model, ps, st = create_model(NeuralODE)

model_stateful, ps_stateful, st_stateful = create_model(StatefulNeuralODE)

x = gpu_device()(ones(Float32, 28, 28, 1, 3));

# NeuralODE is not type stable due to the boxing of `st`

@code_warntype model(x, ps, st)

# We avoid the problem entirely by using `StatefulNeuralODE`

@code_warntype model_stateful(x, ps_stateful, st_stateful)

# Note, that we still recommend using this layer internally and not exposing this as the
# default API to the users.

# Finally checking the compact model

model_compact, ps_compact, st_compact = create_model(NeuralODECompact)

@code_warntype model_compact(x, ps_compact, st_compact)
