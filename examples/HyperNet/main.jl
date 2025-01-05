# # Training a HyperNetwork on MNIST and FashionMNIST

# ## Package Imports

using Lux, ComponentArrays, MLDatasets, MLUtils, OneHotArrays, Optimisers, Printf, Random,
      Reactant

# ## Loading Datasets
function load_dataset(
        ::Type{dset}, n_train::Union{Nothing, Int},
        n_eval::Union{Nothing, Int}, batchsize::Int
) where {dset}
    data = dset(:train)
    (imgs, labels) = if n_train === nothing
        n_train = size(data.features, ndims(data.features))
        data.features, data.targets
    else
        data = data[1:n_train]
        data.features, data.targets
    end
    x_train, y_train = reshape(imgs, 28, 28, 1, n_train), onehotbatch(labels, 0:9)

    data = dset(:test)
    (imgs, labels) = if n_eval === nothing
        n_eval = size(data.features, ndims(data.features))
        data.features, data.targets
    else
        data = data[1:n_eval]
        data.features, data.targets
    end
    x_test, y_test = reshape(imgs, 28, 28, 1, n_eval), onehotbatch(labels, 0:9)

    return (
        DataLoader(
            (x_train, y_train); batchsize=min(batchsize, n_train), shuffle=true,
            partial=false
        ),
        DataLoader(
            (x_test, y_test); batchsize=min(batchsize, n_eval), shuffle=false,
            partial=false
        )
    )
end

function load_datasets(batchsize=256)
    n_train = parse(Bool, get(ENV, "CI", "false")) ? 1024 : nothing
    n_eval = parse(Bool, get(ENV, "CI", "false")) ? 32 : nothing
    return load_dataset.((MNIST, FashionMNIST), n_train, n_eval, batchsize)
end

# ## Implement a HyperNet Layer
function HyperNet(weight_generator::AbstractLuxLayer, core_network::AbstractLuxLayer)
    ca_axes = Lux.initialparameters(Random.default_rng(), core_network) |>
              ComponentArray |> getaxes
    return @compact(; ca_axes, weight_generator, core_network, dispatch=:HyperNet) do (x, y)
        ## Generate the weights
        ps_new = ComponentArray(Lux.Utils.vec(weight_generator(x)), ca_axes)
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
        Embedding(1 => 32),
        Dense(32, 64, relu),
        Dense(64, Lux.parameterlength(core_network))
    )
    return HyperNet(weight_generator, core_network)
end

# ## Define Utility Functions
const loss = CrossEntropyLoss(; logits=Val(true))

function accuracy(model, ps, st, dataloader, idx)
    total_correct, total = 0, 0
    st = Lux.testmode(st)
    for (x, y) in dataloader
        target_class = onecold(y)
        predicted_class = onecold(Array(first(model((idx, x), ps, st))))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end

# ## Training
function train(; dev = reactant_device())
    model = create_model()
    dataloaders = load_datasets(256) |> dev

    ps, st = Lux.setup(Random.default_rng(), model) |> dev

    if dev isa ReactantDevice
        idx = ConcreteRNumber(1)
        x = dev(rand(Float32, 28, 28, 1, 256))
        model_compiled = @compile model((idx, x), ps, Lux.testmode(st))
    else
        model_compiled = model
    end

    train_state = Training.TrainState(model, ps, st, Adam(0.001f0))

    ### Lets train the model
    nepochs = 50
    for epoch in 1:nepochs, data_idx in 1:2
        train_dataloader, test_dataloader = dataloaders[data_idx]
        idx = dev isa ReactantDevice ? ConcreteRNumber(data_idx) : data_idx

        stime = time()
        for (x, y) in train_dataloader
            (_, _, _, train_state) = Training.single_train_step!(
                AutoEnzyme(), loss, ((idx, x), y), train_state
            )
        end
        ttime = time() - stime

        train_acc = round(
            accuracy(
                model_compiled, train_state.parameters,
                train_state.states, train_dataloader, idx
            ) * 100;
            digits=2
        )
        test_acc = round(
            accuracy(
                model_compiled, train_state.parameters,
                train_state.states, test_dataloader, idx
            ) * 100;
            digits=2
        )

        data_name = data_idx == 1 ? "MNIST" : "FashionMNIST"

        @printf "[%3d/%3d]\t%12s\tTime %3.5fs\tTraining Accuracy: %3.2f%%\tTest \
                 Accuracy: %3.2f%%\n" epoch nepochs data_name ttime train_acc test_acc
    end

    println()

    test_acc_list = [0.0, 0.0]
    for data_idx in 1:2
        train_dataloader, test_dataloader = dataloaders[data_idx]
        idx = dev isa ReactantDevice ? ConcreteRNumber(data_idx) : data_idx

        train_acc = round(
            accuracy(
                model_compiled, train_state.parameters,
                train_state.states, train_dataloader, idx
            ) * 100;
            digits=2
        )
        test_acc = round(
            accuracy(
                model_compiled, train_state.parameters,
                train_state.states, test_dataloader, idx
            ) * 100;
            digits=2
        )

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
