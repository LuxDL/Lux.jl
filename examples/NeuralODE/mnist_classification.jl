# MNIST Classification using Neural ODEs
## Package Imports
using Lux,
    DiffEqSensitivity, OrdinaryDiffEq, Random, CUDA, MLDataUtils, Printf, MLDatasets, Optimisers, ComponentArrays
using Flux.Losses: logitcrossentropy
using Flux.Data: DataLoader
import Flux
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
struct NeuralODE{M<:Lux.AbstractExplicitLayer,So,Se,T,K} <: Lux.AbstractExplicitContainerLayer{(:model,)}
    model::M
    solver::So
    sensealg::Se
    tspan::T
    kwargs::K
end

function NeuralODE(
    model::Lux.AbstractExplicitLayer;
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

diffeqsol_to_array(x::ODESolution{T,N,<:AbstractVector{<:CuArray}}) where {T,N} = dropdims(Lux.gpu(x); dims=3)
diffeqsol_to_array(x::ODESolution) = dropdims(Array(x); dims=3)

function train()
    ## Construct the Neural ODE Model
    model = Chain(
        FlattenLayer(),
        Dense(784, 20, tanh),
        NeuralODE(
            Chain(Dense(20, 10, tanh), Dense(10, 10, tanh), Dense(10, 20, tanh));
            save_everystep=false,
            reltol=1.0f-3,
            abstol=1.0f-3,
            save_start=false,
        ),
        diffeqsol_to_array,
        Dense(20, 10),
    )

    ps, st = setup(MersenneTwister(0), model)
    ps = ComponentArray(ps) |> gpu
    st = st |> gpu

    ## Utility Functions
    get_class(x) = argmax.(eachcol(x))

    function loss(x, y, model, ps, st)
        ŷ, st = model(x, ps, st)
        return logitcrossentropy(ŷ, y), st
    end

    function accuracy(model, ps, st, dataloader)
        total_correct, total = 0, 0
        st = Lux.testmode(st)
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
            "[$epoch/$nepochs] \t Time $(round(ttime; digits=2))s \t Training Accuracy: " *
            "$(round(accuracy(model, ps, st, train_dataloader) * 100; digits=2))% \t " *
            "Test Accuracy: $(round(accuracy(model, ps, st, test_dataloader) * 100; digits=2))%"
        )
    end
end

train()

# [1/10] 	Time 23.27s 	Training Accuracy: 90.88% 	Test Accuracy: 90.45%
# [2/10] 	Time 26.2s 	Training Accuracy: 92.27% 	Test Accuracy: 91.78%
# [3/10] 	Time 26.49s 	Training Accuracy: 93.03% 	Test Accuracy: 92.58%
# [4/10] 	Time 25.94s 	Training Accuracy: 93.57% 	Test Accuracy: 92.8%
# [5/10] 	Time 26.86s 	Training Accuracy: 93.76% 	Test Accuracy: 93.18%
# [6/10] 	Time 26.63s 	Training Accuracy: 94.17% 	Test Accuracy: 93.48%
# [7/10] 	Time 25.39s 	Training Accuracy: 94.41% 	Test Accuracy: 93.72%
# [8/10] 	Time 26.17s 	Training Accuracy: 94.68% 	Test Accuracy: 93.73%
# [9/10] 	Time 27.03s 	Training Accuracy: 94.78% 	Test Accuracy: 93.65%
# [10/10] 	Time 26.04s 	Training Accuracy: 94.97% 	Test Accuracy: 94.02%