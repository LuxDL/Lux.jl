# # MNIST Classification with SimpleChains

# SimpleChains.jl is an excellent framework for training small neural networks. In this
# tutorial we will demonstrate how to use the same API as Lux.jl to train a model using
# SimpleChains.jl. We will use the tutorial from
# [SimpleChains.jl](https://pumasai.github.io/SimpleChains.jl/dev/examples/mnist/) as a
# reference.

# ## Package Imports
import Pkg #hide
__DIR = @__DIR__ #hide
pkg_io = open(joinpath(__DIR, "pkg.log"), "w") #hide
Pkg.activate(__DIR; io=pkg_io) #hide
Pkg.instantiate(; io=pkg_io) #hide
Pkg.develop(; path=joinpath(__DIR, "..", ".."), io=pkg_io) #hide
Pkg.precompile(; io=pkg_io) #hide
close(pkg_io) #hide
using Lux, MLUtils, Optimisers, Zygote, OneHotArrays, Random, Statistics
import MLDatasets: MNIST
import SimpleChains: static

# ## Loading MNIST
function loadmnist(batchsize, train_split)
    ## Load MNIST
    N = 2000
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

# ## Define the Model

lux_model = Chain(Conv((5, 5), 1 => 6, relu), MaxPool((2, 2)),
    Conv((5, 5), 6 => 16, relu), MaxPool((2, 2)), FlattenLayer(3),
    Chain(Dense(256 => 128, relu), Dense(128 => 84, relu), Dense(84 => 10)))

# We now need to convert the lux_model to SimpleChains.jl. We need to do this by defining
# the [`ToSimpleChainsAdaptor`](@ref) and providing the input dimensions.

adaptor = ToSimpleChainsAdaptor((static(28), static(28), static(1)))
simple_chains_model = adaptor(lux_model)

# ## Helper Functions
logitcrossentropy(y_pred, y) = mean(-sum(y .* logsoftmax(y_pred); dims=1))

function loss(x, y, model, ps, st)
    y_pred, st = model(x, ps, st)
    return logitcrossentropy(y_pred, y), st
end

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
    ps, st = Lux.setup(rng, model)

    train_dataloader, test_dataloader = loadmnist(128, 0.9)
    opt = Adam(3.0f-4)
    st_opt = Optimisers.setup(opt, ps)

    ### Warmup the Model
    img = train_dataloader.data[1][:, :, :, 1:1]
    lab = train_dataloader.data[2][:, 1:1]
    loss(img, lab, model, ps, st)
    (l, _), back = pullback(p -> loss(img, lab, model, p, st), ps)
    back((one(l), nothing))

    ### Lets train the model
    nepochs = 9
    for epoch in 1:nepochs
        stime = time()
        for (x, y) in train_dataloader
            (l, st), back = pullback(loss, x, y, model, ps, st)
            ### We need to add `nothing`s equal to the number of returned values - 1
            gs = back((one(l), nothing))[4]
            st_opt, ps = Optimisers.update(st_opt, ps, gs)
        end
        ttime = time() - stime

        println("[$epoch/$nepochs] \t Time $(round(ttime; digits=2))s \t Training Accuracy: " *
                "$(round(accuracy(model, ps, st, train_dataloader) * 100; digits=2))% \t " *
                "Test Accuracy: $(round(accuracy(model, ps, st, test_dataloader) * 100; digits=2))%")
    end
end

# ## Finally Training the Model

# First we will train the Lux model
train(lux_model)
nothing #hide

# Now we will train the SimpleChains model
train(simple_chains_model)
nothing #hide

# On my local machine we see a 3-4x speedup when using SimpleChains.jl.
