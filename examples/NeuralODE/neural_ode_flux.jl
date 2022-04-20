# MNIST Classification using Neural ODEs
## Package Imports
using DiffEqSensitivity, OrdinaryDiffEq, Random, Flux, CUDA, MLDataUtils, Printf, MLDatasets, Optimisers
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
struct NeuralODEFlux{M,P,So,Se,T,K}
    model::M
    ps::P
    solver::So
    sensealg::Se
    tspan::T
    kwargs::K
end

Flux.@functor NeuralODEFlux

function NeuralODEFlux(
    model;
    solver=Tsit5(),
    sensealg=InterpolatingAdjoint(; autojacvec=ZygoteVJP()),
    tspan=(0.0f0, 1.0f0),
    kwargs...,
)
    p, re = Flux.destructure(model)
    return NeuralODEFlux(re, p, solver, sensealg, tspan, kwargs)
end

function (n::NeuralODEFlux)(x)
    dudt(u, p, t) =  n.model(p)(u)
    prob = ODEProblem{false}(ODEFunction{false}(dudt), x, n.tspan, n.ps)
    return solve(prob, n.solver; sensealg=n.sensealg, n.kwargs...)
end

diffeqsol_to_array(x::ODESolution{T,N,<:AbstractVector{<:CuArray}}) where {T,N} = dropdims(gpu(x); dims=3)
diffeqsol_to_array(x::ODESolution) = dropdims(Array(x); dims=3)

## Construct the Neural ODE Model
model = Chain(
    Flux.flatten,
    Dense(784, 20, tanh),
    NeuralODEFlux(
        Chain(Dense(20, 10, tanh), Dense(10, 10, tanh), Dense(10, 20, tanh));
        save_everystep=false,
        reltol=1.0f-3,
        abstol=1.0f-3,
        save_start=false,
    ),
    diffeqsol_to_array,
    Dense(20, 10),
) |> gpu

## Utility Functions
get_class(x) = argmax.(eachcol(x))

loss(x, y, model) = logitcrossentropy(model(x), y)

function accuracy(model, dataloader)
    total_correct, total = 0, 0
    for (x, y) in CuIterator(dataloader)
        target_class = get_class(cpu(y))
        predicted_class = get_class(cpu(model(x)))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end

## Training
train_dataloader, test_dataloader = loadmnist(128, 0.9)

ps = Flux.params(model)
opt = Optimisers.ADAM(0.001f0)
st_opt = IdDict()
for p in ps
    st_opt[p] = Optimisers.setup(opt, p)
end

### Warmup the Model
img, lab = gpu(train_dataloader.data[1][:, :, :, 1:1]), gpu(train_dataloader.data[2][:, 1:1])
loss(img, lab, model)
l, back = Flux.pullback(() -> loss(img, lab, model), ps)
back(one(l))

### Lets train the model
nepochs = 10
for epoch in 1:nepochs
    stime = time()
    for (x, y) in CuIterator(train_dataloader)
        l, back = Flux.pullback(() -> loss(x, y, model), ps)
        gs = back(one(l))
        for p in ps
            Optimisers.update!(st_opt[p], p, gs[p])
        end
    end
    ttime = time() - stime

    println(
        "[$epoch/$nepochs] \t Time $(ttime)s \t Training Accuracy: $(accuracy(model, train_dataloader) * 100)% \t Test Accuracy: $(accuracy(model, test_dataloader) * 100)%",
    )
end

# [1/10] 	Time 20.796339988708496s 	Training Accuracy: 91.42037037037038% 	Test Accuracy: 90.68333333333334%
# [2/10] 	Time 21.688892126083374s 	Training Accuracy: 92.75925925925927% 	Test Accuracy: 91.56666666666666%
# [3/10] 	Time 22.588804006576538s 	Training Accuracy: 93.70555555555555% 	Test Accuracy: 92.65%
# [4/10] 	Time 22.03611183166504s 	Training Accuracy: 94.32777777777778% 	Test Accuracy: 93.55%
# [5/10] 	Time 22.306117057800293s 	Training Accuracy: 94.56296296296296% 	Test Accuracy: 93.46666666666667%
# [6/10] 	Time 22.130173921585083s 	Training Accuracy: 94.95185185185186% 	Test Accuracy: 93.83333333333333%
# [7/10] 	Time 21.25230598449707s 	Training Accuracy: 95.19444444444444% 	Test Accuracy: 93.96666666666667%
# [8/10] 	Time 20.79810404777527s 	Training Accuracy: 95.50740740740741% 	Test Accuracy: 94.25%
# [9/10] 	Time 20.77591609954834s 	Training Accuracy: 95.62037037037037% 	Test Accuracy: 94.36666666666666%
# [10/10] 	Time 22.60972309112549s 	Training Accuracy: 95.75% 	Test Accuracy: 94.23333333333333%