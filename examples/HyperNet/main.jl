# # Training a HyperNetwork on MNIST and FashionMNIST

# ## Package Imports

using Lux, ADTypes, ComponentArrays, AMDGPU, LuxCUDA, MLDatasets, MLUtils, OneHotArrays,
      Optimisers, Printf, Random, Setfield, Statistics, Zygote

CUDA.allowscalar(false)

# ## Loading Datasets
function load_dataset(::Type{dset}, n_train::Int, n_eval::Int, batchsize::Int) where {dset}
    imgs, labels = dset(:train)[1:n_train]
    x_train, y_train = reshape(imgs, 28, 28, 1, n_train), onehotbatch(labels, 0:9)

    imgs, labels = dset(:test)[1:n_eval]
    x_test, y_test = reshape(imgs, 28, 28, 1, n_eval), onehotbatch(labels, 0:9)

    return (DataLoader((x_train, y_train); batchsize=min(batchsize, n_train), shuffle=true),
        DataLoader((x_test, y_test); batchsize=min(batchsize, n_eval), shuffle=false))
end

function load_datasets(n_train=1024, n_eval=32, batchsize=256)
    return load_dataset.((MNIST, FashionMNIST), n_train, n_eval, batchsize)
end

# ## Implement a HyperNet Layer
function HyperNet(weight_generator::Lux.AbstractExplicitLayer,
        core_network::Lux.AbstractExplicitLayer)
    ca_axes = Lux.initialparameters(Random.default_rng(), core_network) |>
              ComponentArray |>
              getaxes
    return @compact(; ca_axes, weight_generator, core_network, dispatch=:HyperNet) do (x, y)
        ## Generate the weights
        ps_new = ComponentArray(vec(weight_generator(x)), ca_axes)
        @return core_network(y, ps_new)
    end
end

# Defining functions on the CompactLuxLayer requires some understanding of how the layer
# is structured, as such we don't recommend doing it unless you are familiar with the
# internals. In this case, we simply write it to ignore the initialization of the 
# `core_network` parameters.

function Lux.initialparameters(rng::AbstractRNG, hn::CompactLuxLayer{:HyperNet})
    return (; weight_generator=Lux.initialparameters(rng, hn.layers.weight_generator),)
end

# ## Create and Initialize the HyperNet
function create_model()
    ## Doesn't need to be a MLP can have any Lux Layer
    core_network = Chain(FlattenLayer(), Dense(784, 256, relu), Dense(256, 10))
    weight_generator = Chain(Embedding(2 => 32), Dense(32, 64, relu),
        Dense(64, Lux.parameterlength(core_network)))

    model = HyperNet(weight_generator, core_network)
    return model
end

# ## Define Utility Functions
const loss = CrossEntropyLoss(; logits=Val(true))

function accuracy(model, ps, st, dataloader, data_idx, gdev=gpu_device())
    total_correct, total = 0, 0
    st = Lux.testmode(st)
    cpu_dev = cpu_device()
    for (x, y) in dataloader
        x = x |> gdev
        y = y |> gdev
        target_class = onecold(cpu_dev(y))
        predicted_class = onecold(cpu_dev(model((data_idx, x), ps, st)[1]))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end

# ## Training
function train()
    model = create_model()
    dataloaders = load_datasets()

    dev = gpu_device()
    rng = Xoshiro(0)
    ps, st = Lux.setup(rng, model) |> dev

    train_state = Training.TrainState(model, ps, st, Adam(3.0f-4))

    ### Lets train the model
    nepochs = 10
    for epoch in 1:nepochs, data_idx in 1:2
        train_dataloader, test_dataloader = dataloaders[data_idx]

        stime = time()
        for (x, y) in train_dataloader
            x = x |> dev
            y = y |> dev
            (_, _, _, train_state) = Training.single_train_step!(
                AutoZygote(), loss, ((data_idx, x), y), train_state)
        end
        ttime = time() - stime

        train_acc = round(
            accuracy(model, train_state.parameters, train_state.states,
                train_dataloader, data_idx, dev) * 100;
            digits=2)
        test_acc = round(
            accuracy(model, train_state.parameters, train_state.states,
                test_dataloader, data_idx, dev) * 100;
            digits=2)

        data_name = data_idx == 1 ? "MNIST" : "FashionMNIST"

        @printf "[%3d/%3d] \t %12s \t Time %.5fs \t Training Accuracy: %.2f%% \t Test \
                 Accuracy: %.2f%%\n" epoch nepochs data_name ttime train_acc test_acc
    end

    println()

    for data_idx in 1:2
        train_dataloader, test_dataloader = dataloaders[data_idx]
        train_acc = round(
            accuracy(model, train_state.parameters, train_state.states,
                train_dataloader, data_idx, dev) * 100;
            digits=2)
        test_acc = round(
            accuracy(model, train_state.parameters, train_state.states,
                test_dataloader, data_idx, dev) * 100;
            digits=2)

        data_name = data_idx == 1 ? "MNIST" : "FashionMNIST"

        @printf "[FINAL] \t %12s \t Training Accuracy: %.2f%% \t Test Accuracy: \
                 %.2f%%\n" data_name train_acc test_acc
    end
end

train()
