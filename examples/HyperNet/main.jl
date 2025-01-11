# # Training a HyperNetwork on MNIST and FashionMNIST

# ## Package Imports

using Lux, ComponentArrays, LuxCUDA, MLDatasets, MLUtils, OneHotArrays, Optimisers,
      Printf, Random, Zygote

CUDA.allowscalar(false)

# ## Loading Datasets
function load_dataset(::Type{dset}, n_train::Union{Nothing, Int},
        n_eval::Union{Nothing, Int}, batchsize::Int) where {dset}
    (; features, targets) = if n_train === nothing
        tmp = dset(:train)
        tmp[1:length(tmp)]
    else
        dset(:train)[1:n_train]
    end
    x_train, y_train = reshape(features, 28, 28, 1, :), onehotbatch(targets, 0:9)

    (; features, targets) = if n_eval === nothing
        tmp = dset(:test)
        tmp[1:length(tmp)]
    else
        dset(:test)[1:n_eval]
    end
    x_test, y_test = reshape(features, 28, 28, 1, :), onehotbatch(targets, 0:9)

    return (
        DataLoader(
            (x_train, y_train); batchsize=min(batchsize, size(x_train, 4)), shuffle=true),
        DataLoader(
            (x_test, y_test); batchsize=min(batchsize, size(x_test, 4)), shuffle=false)
    )
end

function load_datasets(batchsize=256)
    n_train = parse(Bool, get(ENV, "CI", "false")) ? 1024 : nothing
    n_eval = parse(Bool, get(ENV, "CI", "false")) ? 32 : nothing
    return load_dataset.((MNIST, FashionMNIST), n_train, n_eval, batchsize)
end

# ## Implement a HyperNet Layer
function HyperNet(
        weight_generator::Lux.AbstractLuxLayer, core_network::Lux.AbstractLuxLayer)
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
    weight_generator = Chain(
        Embedding(2 => 32),
        Dense(32, 64, relu),
        Dense(64, Lux.parameterlength(core_network))
    )

    model = HyperNet(weight_generator, core_network)
    return model
end

# ## Define Utility Functions
const loss = CrossEntropyLoss(; logits=Val(true))

function accuracy(model, ps, st, dataloader, data_idx)
    total_correct, total = 0, 0
    st = Lux.testmode(st)
    for (x, y) in dataloader
        target_class = onecold(y)
        predicted_class = onecold(first(model((data_idx, x), ps, st)))
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

    train_state = Training.TrainState(model, ps, st, Adam(0.001f0))

    ### Lets train the model
    nepochs = 50
    for epoch in 1:nepochs, data_idx in 1:2
        train_dataloader, test_dataloader = dataloaders[data_idx] .|> dev

        stime = time()
        for (x, y) in train_dataloader
            (_, _, _, train_state) = Training.single_train_step!(
                AutoZygote(), loss, ((data_idx, x), y), train_state)
        end
        ttime = time() - stime

        train_acc = round(
            accuracy(model, train_state.parameters,
                train_state.states, train_dataloader, data_idx) * 100;
            digits=2)
        test_acc = round(
            accuracy(model, train_state.parameters,
                train_state.states, test_dataloader, data_idx) * 100;
            digits=2)

        data_name = data_idx == 1 ? "MNIST" : "FashionMNIST"

        @printf "[%3d/%3d]\t%12s\tTime %3.5fs\tTraining Accuracy: %3.2f%%\tTest \
                 Accuracy: %3.2f%%\n" epoch nepochs data_name ttime train_acc test_acc
    end

    println()

    test_acc_list = [0.0, 0.0]
    for data_idx in 1:2
        train_dataloader, test_dataloader = dataloaders[data_idx] .|> dev
        train_acc = round(
            accuracy(model, train_state.parameters,
                train_state.states, train_dataloader, data_idx) * 100;
            digits=2)
        test_acc = round(
            accuracy(model, train_state.parameters,
                train_state.states, test_dataloader, data_idx) * 100;
            digits=2)

        data_name = data_idx == 1 ? "MNIST" : "FashionMNIST"

        @printf "[FINAL]\t%12s\tTraining Accuracy: %3.2f%%\tTest Accuracy: \
                 %3.2f%%\n" data_name train_acc test_acc
        test_acc_list[data_idx] = test_acc
    end
    return test_acc_list
end

test_acc_list = train()
@assert test_acc_list[1] > 60 && test_acc_list[2] > 60 #hide
nothing #hide
