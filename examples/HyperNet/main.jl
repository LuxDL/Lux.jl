# # Training a HyperNetwork on MNIST and FashionMNIST

# ## Package Imports
using Pkg #hide
__DIR = @__DIR__ #hide
pkg_io = open(joinpath(__DIR, "pkg.log"), "w") #hide
Pkg.activate(__DIR; io=pkg_io) #hide
Pkg.instantiate(; io=pkg_io) #hide
Pkg.develop(; path=joinpath(__DIR, "..", ".."), io=pkg_io) #hide
Pkg.precompile(; io=pkg_io) #hide
close(pkg_io) #hide
using Lux, ADTypes, ComponentArrays, LuxAMDGPU, LuxCUDA, MLDatasets, MLUtils, OneHotArrays,
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
struct HyperNet{W <: Lux.AbstractExplicitLayer, C <: Lux.AbstractExplicitLayer, A} <:
       Lux.AbstractExplicitContainerLayer{(:weight_generator, :core_network)}
    weight_generator::W
    core_network::C
    ca_axes::A
end

function HyperNet(w::Lux.AbstractExplicitLayer, c::Lux.AbstractExplicitLayer)
    ca_axes = Lux.initialparameters(Random.default_rng(), c) |> ComponentArray |> getaxes
    return HyperNet(w, c, ca_axes)
end

function Lux.initialparameters(rng::AbstractRNG, h::HyperNet)
    return (weight_generator=Lux.initialparameters(rng, h.weight_generator),)
end

function (hn::HyperNet)(x, ps, st::NamedTuple)
    ps_new, st_ = hn.weight_generator(x, ps.weight_generator, st.weight_generator)
    @set! st.weight_generator = st_
    return ComponentArray(vec(ps_new), hn.ca_axes), st
end

function (hn::HyperNet)((x, y)::T, ps, st::NamedTuple) where {T <: Tuple}
    ps_ca, st = hn(x, ps, st)
    pred, st_ = hn.core_network(y, ps_ca, st.core_network)
    @set! st.core_network = st_
    return pred, st
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
logitcrossentropy(y_pred, y) = mean(-sum(y .* logsoftmax(y_pred); dims=1))

function loss(model, ps, st, (data_idx, x, y))
    y_pred, st = model((data_idx, x), ps, st)
    return logitcrossentropy(y_pred, y), st, (;)
end

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

    train_state = Lux.Experimental.TrainState(
        rng, model, Adam(3.0f-4); transform_variables=dev)

    ### Lets train the model
    nepochs = 10
    for epoch in 1:nepochs, data_idx in 1:2
        train_dataloader, test_dataloader = dataloaders[data_idx]

        stime = time()
        for (x, y) in train_dataloader
            x = x |> dev
            y = y |> dev
            (gs, _, _, train_state) = Lux.Experimental.compute_gradients(
                AutoZygote(), loss, (data_idx, x, y), train_state)
            train_state = Lux.Experimental.apply_gradients(train_state, gs)
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

        @printf "[%3d/%3d] \t %12s \t Time %.2fs \t Training Accuracy: %.2f%% \t Test Accuracy: %.2f%%\n" epoch nepochs data_name ttime train_acc test_acc
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

        @printf "[FINAL] \t %12s \t Training Accuracy: %.2f%% \t Test Accuracy: %.2f%%\n" data_name train_acc test_acc
    end
end

train()
