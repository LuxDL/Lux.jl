# # MNIST Classification with SimpleChains

# SimpleChains.jl is an excellent framework for training small neural networks. In this
# tutorial we will demonstrate how to use the same API as Lux.jl to train a model using
# SimpleChains.jl. We will use the tutorial from
# [SimpleChains.jl](https://pumasai.github.io/SimpleChains.jl/dev/examples/mnist/) as a
# reference.

# ## Package Imports
using Lux, ADTypes, MLUtils, Optimisers, Zygote, OneHotArrays, Random, Statistics, Printf
using MLDatasets: MNIST
using SimpleChains: SimpleChains

# ## Loading MNIST
function loadmnist(batchsize, train_split)
    ## Load MNIST
    N = 2000
    dataset = MNIST(; split=:train)
    imgs = dataset.features[:, :, 1:N]
    labels_raw = dataset.targets[1:N]

    ## Process images into (H, W, C, BS) batches
    x_data = Float32.(reshape(imgs, size(imgs, 1), size(imgs, 2), 1, size(imgs, 3)))
    y_data = onehotbatch(labels_raw, 0:9)
    (x_train, y_train), (x_test, y_test) = splitobs((x_data, y_data); at=train_split)

    return (
        ## Use DataLoader to automatically minibatch and shuffle the data
        DataLoader(collect.((x_train, y_train)); batchsize, shuffle=true),
        ## Don't shuffle the test data
        DataLoader(collect.((x_test, y_test)); batchsize, shuffle=false))
end

# ## Define the Model

lux_model = Chain(Conv((5, 5), 1 => 6, relu), MaxPool((2, 2)),
    Conv((5, 5), 6 => 16, relu), MaxPool((2, 2)), FlattenLayer(3),
    Chain(Dense(256 => 128, relu), Dense(128 => 84, relu), Dense(84 => 10)))

# We now need to convert the lux_model to SimpleChains.jl. We need to do this by defining
# the [`ToSimpleChainsAdaptor`](@ref) and providing the input dimensions.

adaptor = ToSimpleChainsAdaptor((28, 28, 1))
simple_chains_model = adaptor(lux_model)

# ## Helper Functions
const loss = CrossEntropyLoss(; logits=Val(true))

function accuracy(model, ps, st, dataloader)
    total_correct, total = 0, 0
    st = Lux.testmode(st)
    for (x, y) in dataloader
        target_class = onecold(y)
        predicted_class = onecold(Array(first(model(x, ps, st))))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end

# ## Define the Training Loop
function train(model; rng=Xoshiro(0), kwargs...)
    train_dataloader, test_dataloader = loadmnist(128, 0.9)
    ps, st = Lux.setup(rng, model)

    train_state = Training.TrainState(model, ps, st, Adam(3.0f-4))

    ### Warmup the model
    x_proto = randn(rng, Float32, 28, 28, 1, 1)
    y_proto = onehotbatch([1], 0:9)
    Training.compute_gradients(AutoZygote(), loss, (x_proto, y_proto), train_state)

    ### Lets train the model
    nepochs = 10
    tr_acc, te_acc = 0.0, 0.0
    for epoch in 1:nepochs
        stime = time()
        for (x, y) in train_dataloader
            gs, _,
            _,
            train_state = Training.single_train_step!(
                AutoZygote(), loss, (x, y), train_state)
        end
        ttime = time() - stime

        tr_acc = accuracy(
            model, train_state.parameters, train_state.states, train_dataloader) * 100
        te_acc = accuracy(
            model, train_state.parameters, train_state.states, test_dataloader) * 100

        @printf "[%2d/%2d] \t Time %.2fs \t Training Accuracy: %.2f%% \t Test Accuracy: \
                 %.2f%%\n" epoch nepochs ttime tr_acc te_acc
    end

    return tr_acc, te_acc
end

# ## Finally Training the Model

# First we will train the Lux model
tr_acc, te_acc = train(lux_model)
@assert tr_acc > 0.75 && te_acc > 0.75 #hide
nothing #hide

# Now we will train the SimpleChains model
train(simple_chains_model)
@assert tr_acc > 0.75 && te_acc > 0.75 #hide
nothing #hide

# On my local machine we see a 3-4x speedup when using SimpleChains.jl. The conditions of
# the server this documentation is being built on is not ideal for CPU benchmarking hence,
# the speedup may not be as significant and even there might be regressions.
