# ## Package Imports
using Lux
using Pkg #hide
Pkg.activate(joinpath(dirname(pathof(Lux)), "..", "examples")) #hide
using ComponentArrays, CUDA, MLDatasets, MLUtils, NNlib, OneHotArrays, Optimisers, Random,
      Setfield, Statistics, Zygote
CUDA.allowscalar(false)

# ## Loading Datasets
function _load_dataset(dset, n_train::Int, n_eval::Int, batchsize::Int)
    imgs, labels = dset(:train)[1:n_train]
    x_train, y_train = reshape(imgs, 28, 28, 1, n_train), onehotbatch(labels, 0:9)

    imgs, labels = dset(:test)[1:n_eval]
    x_test, y_test = reshape(imgs, 28, 28, 1, n_eval), onehotbatch(labels, 0:9)

    return (DataLoader((x_train, y_train); batchsize=min(batchsize, n_train), shuffle=true),
            DataLoader((x_test, y_test); batchsize=min(batchsize, n_eval), shuffle=false))
end

function load_datasets(n_train=1024, n_eval=32, batchsize=256)
    return _load_dataset.((MNIST, FashionMNIST), n_train, n_eval, batchsize)
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

function (hn::HyperNet)(x, ps, st::NamedTuple) where {T <: Tuple}
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

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    ps, st = Lux.setup(rng, model) .|> gpu

    return model, ps, st
end

# ## Define Utility Functions
logitcrossentropy(y_pred, y) = mean(-sum(y .* logsoftmax(y_pred); dims=1))

function loss(data_idx, x, y, model, ps, st)
    y_pred, st = model((data_idx, x), ps, st)
    return logitcrossentropy(y_pred, y), st
end

function accuracy(model, ps, st, dataloader, data_idx)
    total_correct, total = 0, 0
    st = Lux.testmode(st)
    for (x, y) in dataloader
        x = x |> gpu
        y = y |> gpu
        target_class = onecold(cpu(y))
        predicted_class = onecold(cpu(model((data_idx, x), ps, st)[1]))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end

# ## Training
function train()
    model, ps, st = create_model()

    ## Training
    dataloaders = load_datasets()

    opt = Adam(0.001f0)
    st_opt = Optimisers.setup(opt, ps)

    ### Warmup the Model
    img, lab = gpu(dataloaders[1][1].data[1][:, :, :, 1:1]),
               gpu(dataloaders[1][1].data[2][:, 1:1])
    loss(1, img, lab, model, ps, st)
    (l, _), back = pullback(p -> loss(1, img, lab, model, p, st), ps)
    back((one(l), nothing))

    ### Lets train the model
    nepochs = 9
    for epoch in 1:nepochs
        for data_idx in 1:2
            train_dataloader, test_dataloader = dataloaders[data_idx]

            stime = time()
            for (x, y) in train_dataloader
                x = x |> gpu
                y = y |> gpu
                (l, st), back = pullback(p -> loss(data_idx, x, y, model, p, st), ps)
                gs = back((one(l), nothing))[1]
                st_opt, ps = Optimisers.update(st_opt, ps, gs)
            end
            ttime = time() - stime

            train_acc = round(accuracy(model, ps, st, train_dataloader, data_idx) * 100;
                              digits=2)
            test_acc = round(accuracy(model, ps, st, test_dataloader, data_idx) * 100;
                             digits=2)

            data_name = data_idx == 1 ? "MNIST" : "FashionMNIST"

            println("[$epoch/$nepochs] \t $data_name Time $(round(ttime; digits=2))s \t " *
                    "Training Accuracy: $(train_acc)% \t Test Accuracy: $(test_acc)%")
        end
    end

    for data_idx in 1:2
        train_dataloader, test_dataloader = dataloaders[data_idx]
        train_acc = round(accuracy(model, ps, st, train_dataloader, data_idx) * 100;
                          digits=2)
        test_acc = round(accuracy(model, ps, st, test_dataloader, data_idx) * 100; digits=2)

        data_name = data_idx == 1 ? "MNIST" : "FashionMNIST"

        println("[FINAL] \t $data_name Training Accuracy: $(train_acc)% \t " *
                "Test Accuracy: $(test_acc)%")
    end
end

train()
