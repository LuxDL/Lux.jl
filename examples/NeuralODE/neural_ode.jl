# MNIST Classification using Neural ODEs
## Package Imports
using ExplicitFluxLayers,
    DiffEqSensitivity, OrdinaryDiffEq, Random, Flux, CUDA, MLDataUtils, Printf, MLDatasets, Optimisers
using Flux.Losses: logitcrossentropy
using Flux.Data: DataLoader
CUDA.allowscalar(false)

## DataLoader
function loadmnist(batchsize, train_split)
    # Use MLDataUtils LabelEnc for natural onehot conversion
    onehot(labels_raw) = convertlabel(LabelEnc.OneOfK, labels_raw, LabelEnc.NativeLabels(collect(0:9)))
    # Load MNIST
    imgs, labels_raw = MNIST.traindata()
    # Process images into (H,W,C,BS) batches
    x_data = Float32.(reshape(imgs, size(imgs, 1), size(imgs, 2), 1, size(imgs, 3)))
    y_data = onehot(labels_raw)
    (x_train, y_train), (x_test, y_test) = stratifiedobs((x_data, y_data); p=train_split)
    return (
        # Use Flux's DataLoader to automatically minibatch and shuffle the data
        DataLoader(collect.((x_train, y_train)); batchsize=batchsize, shuffle=true),
        # Don't shuffle the test data
        DataLoader(collect.((x_test, y_test)); batchsize=batchsize, shuffle=false),
    )
end

## Define the Neural ODE Layer
struct NeuralODE{M<:EFL.AbstractExplicitLayer,So,Se,T,K} <: EFL.AbstractExplicitContainerLayer{(:model,)}
    model::M
    solver::So
    sensealg::Se
    tspan::T
    kwargs::K
end

function NeuralODE(
    model::EFL.AbstractExplicitLayer;
    solver=Tsit5(),
    sensealg=InterpolatingAdjoint(; autojacvec=ZygoteVJP()),
    tspan=(0.0f0, 1.0f0),
    kwargs...,
)
    return NeuralODE(model, solver, sensealg, tspan, kwargs)
end

function (n::NeuralODE)(x, ps, st)
    function dudt(u, p, t)
        u_, st = n.model(u, p, st)
        return u_
    end
    prob = ODEProblem{false}(ODEFunction{false}(dudt), x, n.tspan, ps)
    return solve(prob, n.solver; sensealg=n.sensealg, n.kwargs...), st
end

diffeqsol_to_array(x::ODESolution{T,N,<:AbstractVector{<:CuArray}}) where {T,N} = dropdims(gpu(x); dims=3)
diffeqsol_to_array(x::ODESolution) = dropdims(Array(x); dims=3)

## Construct the Neural ODE Model
model = EFL.Chain(
    EFL.FlattenLayer(),
    EFL.Dense(784, 20, tanh),
    NeuralODE(
        EFL.Chain(EFL.Dense(20, 10, tanh), EFL.Dense(10, 10, tanh), EFL.Dense(10, 20, tanh));
        save_everystep=false,
        reltol=1.0f-3,
        abstol=1.0f-3,
        save_start=false,
    ),
    diffeqsol_to_array,
    EFL.Dense(20, 10),
)

ps, st = gpu.(EFL.setup(MersenneTwister(0), model))

## Utility Functions
get_class(x) = argmax.(eachcol(x))

function loss(x, y, model, ps, st)
    ŷ, st = model(x, ps, st)
    return logitcrossentropy(ŷ, y), st
end

function accuracy(model, ps, st, dataloader)
    total_correct, total = 0, 0
    st = EFL.testmode(st)
    for (x, y) in CuIterator(dataloader)
        target_class = get_class(cpu(y))
        predicted_class = get_class(cpu(model(x, ps, st)[1]))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end

## Training
train_dataloader, test_dataloader = loadmnist(128, 0.9)

opt = Optimisers.ADAM(0.001f0)
st_opt = Optimisers.setup(opt, ps)

### Warmup the Model
img, lab = gpu(train_dataloader.data[1][:, :, :, 1:1]), gpu(train_dataloader.data[2][:, 1:1])
loss(img, lab, model, ps, st)
(l, _), back = Flux.pullback(p -> loss(img, lab, model, p, st), ps)
back((one(l), nothing))

### Lets train the model
nepochs = 10
for epoch in 1:nepochs
    stime = time()
    for (x, y) in CuIterator(train_dataloader)
        (l, _), back = Flux.pullback(p -> loss(x, y, model, p, st), ps)
        gs = back((one(l), nothing))[1]
        st_opt, ps = Optimisers.update(st_opt, ps, gs)
    end
    ttime = time() - stime

    println(
        "[$epoch/$nepochs] \t Time $(ttime)s \t Training Accuracy: $(accuracy(model, ps, st, train_dataloader) * 100)% \t Test Accuracy: $(accuracy(model, ps, st, test_dataloader) * 100)%",
    )
end

# Training Result
# [1/10] 	Time 20.142139196395874s 	Training Accuracy: 91.41296296296296% 	Test Accuracy: 91.0%
# [2/10] 	Time 19.797057151794434s 	Training Accuracy: 92.86851851851851% 	Test Accuracy: 92.21666666666667%
# [3/10] 	Time 20.34799098968506s 	Training Accuracy: 93.55% 	Test Accuracy: 92.71666666666667%
# [4/10] 	Time 21.096089124679565s 	Training Accuracy: 94.16481481481482% 	Test Accuracy: 93.13333333333334%
# [5/10] 	Time 21.31805109977722s 	Training Accuracy: 94.58148148148148% 	Test Accuracy: 93.58333333333333%
# [6/10] 	Time 20.36558485031128s 	Training Accuracy: 94.87962962962962% 	Test Accuracy: 93.88333333333333%
# [7/10] 	Time 20.330584049224854s 	Training Accuracy: 95.21851851851851% 	Test Accuracy: 93.66666666666667%
# [8/10] 	Time 20.88755488395691s 	Training Accuracy: 95.22222222222221% 	Test Accuracy: 93.83333333333333%
# [9/10] 	Time 21.74562406539917s 	Training Accuracy: 95.54444444444444% 	Test Accuracy: 94.23333333333333%
# [10/10] 	Time 21.899173974990845s 	Training Accuracy: 95.7962962962963% 	Test Accuracy: 93.88333333333333%
