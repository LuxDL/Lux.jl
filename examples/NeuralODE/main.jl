# # MNIST Classification using Neural ODEs

# To understand Neural ODEs, users should look up
# [these lecture notes](https://book.sciml.ai/notes/11-Differentiable_Programming_and_Neural_Differential_Equations/). We recommend users to directly use
# [DiffEqFlux.jl](https://docs.sciml.ai/DiffEqFlux/stable/), instead of implementing
# Neural ODEs from scratch.

# ## Package Imports
using Pkg #hide
__DIR = @__DIR__ #hide
pkg_io = open(joinpath(__DIR, "pkg.log"), "w") #hide
Pkg.activate(__DIR; io=pkg_io) #hide
Pkg.instantiate(; io=pkg_io) #hide
Pkg.develop(; path=joinpath(__DIR, "..", ".."), io=pkg_io) #hide
Pkg.precompile(; io=pkg_io) #hide
close(pkg_io) #hide
using Lux, ComponentArrays, SciMLSensitivity, LuxAMDGPU, LuxCUDA, Optimisers,
      OrdinaryDiffEq, Random, Statistics, Zygote, OneHotArrays, InteractiveUtils, Printf
import MLDatasets: MNIST
import MLUtils: DataLoader, splitobs

CUDA.allowscalar(false)

# ## Loading MNIST
function loadmnist(batchsize, train_split)
    ## Load MNIST: Only 1500 for demonstration purposes
    N = 1500
    dataset = MNIST(; split=:train)
    imgs = dataset.features[:, :, 1:N]
    labels_raw = dataset.targets[1:N]

    ## Process images into (H,W,C,BS) batches
    x_data = Float32.(reshape(imgs, size(imgs, 1), size(imgs, 2), 1, size(imgs, 3)))
    y_data = onehotbatch(labels_raw, 0:9)
    (x_train, y_train), (x_test, y_test) = splitobs((x_data, y_data); at=train_split)

    return (
        ## Use DataLoader to automatically minibatch and shuffle the data
        DataLoader(collect.((x_train, y_train)); batchsize, shuffle=true),
        ## Don't shuffle the test data
        DataLoader(collect.((x_test, y_test)); batchsize, shuffle=false))
end

# ## Define the Neural ODE Layer
#
# The NeuralODE is a ContainerLayer, which stores a `model`. The parameters and states of
# the NeuralODE are same as those of the underlying model.
struct NeuralODE{M <: Lux.AbstractExplicitLayer, So, T, K} <:
       Lux.AbstractExplicitContainerLayer{(:model,)}
    model::M
    solver::So
    tspan::T
    kwargs::K
end

function NeuralODE(
        model::Lux.AbstractExplicitLayer; solver=Tsit5(), tspan=(0.0f0, 1.0f0), kwargs...)
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
        model_fn(Chain(Dense(20 => 10, tanh), Dense(10 => 10, tanh), Dense(10 => 20, tanh));
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
logitcrossentropy(y_pred, y) = mean(-sum(y .* logsoftmax(y_pred); dims=1))

function loss(x, y, model, ps, st)
    y_pred, st = model(x, ps, st)
    return logitcrossentropy(y_pred, y), st
end

function accuracy(model, ps, st, dataloader; dev=gpu_device())
    total_correct, total = 0, 0
    st = Lux.testmode(st)
    cpu_dev = cpu_device()
    for (x, y) in dataloader
        target_class = onecold(y)
        predicted_class = onecold(cpu_dev(first(model(dev(x), ps, st))))
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
    train_dataloader, test_dataloader = loadmnist(128, 0.9)

    opt = Adam(0.001f0)
    st_opt = Optimisers.setup(opt, ps)

    ### Warmup the Model
    img = dev(train_dataloader.data[1][:, :, :, 1:1])
    lab = dev(train_dataloader.data[2][:, 1:1])
    loss(img, lab, model, ps, st)
    (l, _), back = pullback(p -> loss(img, lab, model, p, st), ps)
    back((one(l), nothing))

    ### Lets train the model
    nepochs = 9
    for epoch in 1:nepochs
        stime = time()
        for (x, y) in train_dataloader
            x = dev(x)
            y = dev(y)
            (l, st), back = pullback(p -> loss(x, y, model, p, st), ps)
            ### We need to add `nothing`s equal to the number of returned values - 1
            gs = back((one(l), nothing))[1]
            st_opt, ps = Optimisers.update(st_opt, ps, gs)
        end
        ttime = time() - stime

        tr_acc = accuracy(model, ps, st, train_dataloader; dev)
        te_acc = accuracy(model, ps, st, test_dataloader; dev)
        @printf "[%d/%d] \t Time %.2fs \t Training Accuracy: %.5f%% \t Test Accuracy: %.5f%%\n" epoch nepochs ttime tr_acc te_acc
    end
end

train(NeuralODE)

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

# Starting `v0.5.5`, Lux provides a `Lux.Experimental.StatefulLuxLayer` which can be used
# to avoid the [`Box`ing of `st`](https://github.com/JuliaLang/julia/issues/15276).
struct StatefulNeuralODE{M <: Lux.AbstractExplicitLayer, So, T, K} <:
       Lux.AbstractExplicitContainerLayer{(:model,)}
    model::M
    solver::So
    tspan::T
    kwargs::K
end

function StatefulNeuralODE(
        model::Lux.AbstractExplicitLayer; solver=Tsit5(), tspan=(0.0f0, 1.0f0), kwargs...)
    return StatefulNeuralODE(model, solver, tspan, kwargs)
end

function (n::StatefulNeuralODE)(x, ps, st)
    st_model = Lux.StatefulLuxLayer(n.model, ps, st)
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
