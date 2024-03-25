


<a id='Package Imports'></a>

## Package Imports


```julia
using Lux
using ComponentArrays, LuxAMDGPU, LuxCUDA, MLDatasets, MLUtils, OneHotArrays,
    Optimisers, Random, Setfield, Statistics, Zygote
CUDA.allowscalar(false)
```


```
  Activating project at `/var/lib/buildkite-agent/builds/gpuci-11/julialang/lux-dot-jl/examples`
```


<a id='Loading Datasets'></a>

## Loading Datasets


```julia
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
```


```
load_datasets (generic function with 4 methods)
```


<a id='Implement a HyperNet Layer'></a>

## Implement a HyperNet Layer


```julia
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
```


<a id='Create and Initialize the HyperNet'></a>

## Create and Initialize the HyperNet


```julia
function create_model()
    # Doesn't need to be a MLP can have any Lux Layer
    core_network = Chain(FlattenLayer(), Dense(784, 256, relu), Dense(256, 10))
    weight_generator = Chain(Embedding(2 => 32),
        Dense(32, 64, relu),
        Dense(64, Lux.parameterlength(core_network)))

    model = HyperNet(weight_generator, core_network)

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    ps, st = Lux.setup(rng, model) .|> gpu_device()

    return model, ps, st
end
```


```
create_model (generic function with 1 method)
```


<a id='Define Utility Functions'></a>

## Define Utility Functions


```julia
logitcrossentropy(y_pred, y) = mean(-sum(y .* logsoftmax(y_pred); dims=1))

function loss(data_idx, x, y, model, ps, st)
    y_pred, st = model((data_idx, x), ps, st)
    return logitcrossentropy(y_pred, y), st
end

function accuracy(model, ps, st, dataloader, data_idx)
    total_correct, total = 0, 0
    st = Lux.testmode(st)
    dev = gpu_device()
    cpu_dev = cpu_device()
    for (x, y) in dataloader
        x = x |> dev
        y = y |> dev
        target_class = onecold(cpu_dev(y))
        predicted_class = onecold(cpu_dev(model((data_idx, x), ps, st)[1]))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end
```


```
accuracy (generic function with 1 method)
```


<a id='Training'></a>

## Training


```julia
function train()
    model, ps, st = create_model()

    # Training
    dataloaders = load_datasets()

    opt = Adam(0.001f0)
    st_opt = Optimisers.setup(opt, ps)

    dev = gpu_device()

    ### Warmup the Model
    img, lab = dev(dataloaders[1][1].data[1][:, :, :, 1:1]),
    dev(dataloaders[1][1].data[2][:, 1:1])
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
                x = x |> dev
                y = y |> dev
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
```


```
[1/9] 	 MNIST Time 5.12s 	 Training Accuracy: 55.18% 	 Test Accuracy: 53.12%
[1/9] 	 FashionMNIST Time 0.04s 	 Training Accuracy: 61.91% 	 Test Accuracy: 56.25%
[2/9] 	 MNIST Time 0.04s 	 Training Accuracy: 68.75% 	 Test Accuracy: 68.75%
[2/9] 	 FashionMNIST Time 0.04s 	 Training Accuracy: 50.98% 	 Test Accuracy: 37.5%
[3/9] 	 MNIST Time 0.03s 	 Training Accuracy: 83.59% 	 Test Accuracy: 84.38%
[3/9] 	 FashionMNIST Time 0.07s 	 Training Accuracy: 60.45% 	 Test Accuracy: 50.0%
[4/9] 	 MNIST Time 0.03s 	 Training Accuracy: 82.52% 	 Test Accuracy: 68.75%
[4/9] 	 FashionMNIST Time 0.03s 	 Training Accuracy: 62.11% 	 Test Accuracy: 56.25%
[5/9] 	 MNIST Time 0.03s 	 Training Accuracy: 88.28% 	 Test Accuracy: 87.5%
[5/9] 	 FashionMNIST Time 0.08s 	 Training Accuracy: 61.82% 	 Test Accuracy: 59.38%
[6/9] 	 MNIST Time 0.03s 	 Training Accuracy: 87.6% 	 Test Accuracy: 81.25%
[6/9] 	 FashionMNIST Time 0.03s 	 Training Accuracy: 67.38% 	 Test Accuracy: 75.0%
[7/9] 	 MNIST Time 0.03s 	 Training Accuracy: 90.82% 	 Test Accuracy: 87.5%
[7/9] 	 FashionMNIST Time 0.07s 	 Training Accuracy: 70.41% 	 Test Accuracy: 75.0%
[8/9] 	 MNIST Time 0.03s 	 Training Accuracy: 92.68% 	 Test Accuracy: 87.5%
[8/9] 	 FashionMNIST Time 0.03s 	 Training Accuracy: 75.78% 	 Test Accuracy: 68.75%
[9/9] 	 MNIST Time 0.03s 	 Training Accuracy: 95.31% 	 Test Accuracy: 90.62%
[9/9] 	 FashionMNIST Time 0.05s 	 Training Accuracy: 73.63% 	 Test Accuracy: 68.75%
[FINAL] 	 MNIST Training Accuracy: 94.73% 	 Test Accuracy: 90.62%
[FINAL] 	 FashionMNIST Training Accuracy: 73.63% 	 Test Accuracy: 68.75%
```


---


*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

