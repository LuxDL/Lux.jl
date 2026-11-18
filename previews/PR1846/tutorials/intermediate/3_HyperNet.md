---
url: /previews/PR1846/tutorials/intermediate/3_HyperNet.md
---

# Training a HyperNetwork on MNIST and FashionMNIST {#Training-a-HyperNetwork-on-MNIST-and-FashionMNIST}

## Package Imports {#Package-Imports}

```julia
using Lux,
    ComponentArrays, MLDatasets, MLUtils, OneHotArrays, Optimisers, Printf, Random, Reactant
```

## Loading Datasets {#Loading-Datasets}

```julia
function load_dataset(
    ::Type{dset}, n_train::Union{Nothing,Int}, n_eval::Union{Nothing,Int}, batchsize::Int
) where {dset}
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
            (x_train, y_train);
            batchsize=min(batchsize, size(x_train, 4)),
            shuffle=true,
            partial=false,
        ),
        DataLoader(
            (x_test, y_test);
            batchsize=min(batchsize, size(x_test, 4)),
            shuffle=false,
            partial=false,
        ),
    )
end

function load_datasets(batchsize=32)
    n_train = parse(Bool, get(ENV, "CI", "false")) ? 1024 : nothing
    n_eval = parse(Bool, get(ENV, "CI", "false")) ? 32 : nothing
    return load_dataset.((MNIST, FashionMNIST), n_train, n_eval, batchsize)
end
```

## Implement a HyperNet Layer {#Implement-a-HyperNet-Layer}

```julia
function HyperNet(weight_generator::AbstractLuxLayer, core_network::AbstractLuxLayer)
    ca_axes = getaxes(
        ComponentArray(Lux.initialparameters(Random.default_rng(), core_network))
    )
    return @compact(; ca_axes, weight_generator, core_network, dispatch=:HyperNet) do (x, y)
        # Generate the weights
        ps_new = ComponentArray(vec(weight_generator(x)), ca_axes)
        @return core_network(y, ps_new)
    end
end
```

Defining functions on the CompactLuxLayer requires some understanding of how the layer is structured, as such we don't recommend doing it unless you are familiar with the internals. In this case, we simply write it to ignore the initialization of the `core_network` parameters.

```julia
function Lux.initialparameters(rng::AbstractRNG, hn::CompactLuxLayer{:HyperNet})
    return (; weight_generator=Lux.initialparameters(rng, hn.layers.weight_generator))
end
```

## Create and Initialize the HyperNet {#Create-and-Initialize-the-HyperNet}

```julia
function create_model()
    core_network = Chain(
        Conv((3, 3), 1 => 16, relu; stride=2),
        Conv((3, 3), 16 => 32, relu; stride=2),
        Conv((3, 3), 32 => 64, relu; stride=2),
        GlobalMeanPool(),
        FlattenLayer(),
        Dense(64, 10),
    )
    return HyperNet(
        Chain(
            Embedding(2 => 32),
            Dense(32, 64, relu),
            Dense(64, Lux.parameterlength(core_network)),
        ),
        core_network,
    )
end
```

## Define Utility Functions {#Define-Utility-Functions}

```julia
function accuracy(model, ps, st, dataloader, data_idx)
    total_correct, total = 0, 0
    cdev = cpu_device()
    st = Lux.testmode(st)
    for (x, y) in dataloader
        ŷ, _ = model((data_idx, x), ps, st)
        target_class = y |> cdev |> onecold
        predicted_class = ŷ |> cdev |> onecold
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end
```

## Training {#Training}

```julia
function train()
    dev = reactant_device(; force=true)

    model = create_model()
    dataloaders = load_datasets() |> dev

    Random.seed!(1234)
    ps, st = Lux.setup(Random.default_rng(), model) |> dev

    train_state = Training.TrainState(model, ps, st, Adam(0.0003f0))

    x = first(first(dataloaders[1][1]))
    data_idx = ConcreteRNumber(1)
    model_compiled = @compile model((data_idx, x), ps, Lux.testmode(st))

    ### Let's train the model
    nepochs = 50
    for epoch in 1:nepochs, data_idx in 1:2
        train_dataloader, test_dataloader = dev.(dataloaders[data_idx])

        ### This allows us to trace the data index, else it will be embedded as a constant
        ### in the IR
        concrete_data_idx = ConcreteRNumber(data_idx)

        stime = time()
        for (x, y) in train_dataloader
            (_, _, _, train_state) = Training.single_train_step!(
                AutoEnzyme(),
                CrossEntropyLoss(; logits=Val(true)),
                ((concrete_data_idx, x), y),
                train_state;
                return_gradients=Val(false),
            )
        end
        ttime = time() - stime

        train_acc = round(
            accuracy(
                model_compiled,
                train_state.parameters,
                train_state.states,
                train_dataloader,
                concrete_data_idx,
            ) * 100;
            digits=2,
        )
        test_acc = round(
            accuracy(
                model_compiled,
                train_state.parameters,
                train_state.states,
                test_dataloader,
                concrete_data_idx,
            ) * 100;
            digits=2,
        )

        data_name = data_idx == 1 ? "MNIST" : "FashionMNIST"

        @printf "[%3d/%3d]\t%12s\tTime %3.5fs\tTraining Accuracy: %3.2f%%\tTest \
                 Accuracy: %3.2f%%\n" epoch nepochs data_name ttime train_acc test_acc
    end

    println()

    test_acc_list = [0.0, 0.0]
    for data_idx in 1:2
        train_dataloader, test_dataloader = dev.(dataloaders[data_idx])

        concrete_data_idx = ConcreteRNumber(data_idx)
        train_acc = round(
            accuracy(
                model_compiled,
                train_state.parameters,
                train_state.states,
                train_dataloader,
                concrete_data_idx,
            ) * 100;
            digits=2,
        )
        test_acc = round(
            accuracy(
                model_compiled,
                train_state.parameters,
                train_state.states,
                test_dataloader,
                concrete_data_idx,
            ) * 100;
            digits=2,
        )

        data_name = data_idx == 1 ? "MNIST" : "FashionMNIST"

        @printf "[FINAL]\t%12s\tTraining Accuracy: %3.2f%%\tTest Accuracy: \
                 %3.2f%%\n" data_name train_acc test_acc
        test_acc_list[data_idx] = test_acc
    end
    return test_acc_list
end

test_acc_list = train()
```

```
┌ Warning: Reactant unsafe_free! is not defined for OneHotArrays.OneHotMatrix{UInt32, Reactant.ConcretePJRTArray{UInt32, 1, 1}}.
└ @ ReactantExt ~/work/Lux.jl/Lux.jl/lib/MLDataDevices/ext/ReactantExt.jl:106
[  1/ 50]	       MNIST	Time 28.72959s	Training Accuracy: 34.57%	Test Accuracy: 37.50%
[  1/ 50]	FashionMNIST	Time 0.06273s	Training Accuracy: 32.62%	Test Accuracy: 43.75%
[  2/ 50]	       MNIST	Time 0.01104s	Training Accuracy: 36.72%	Test Accuracy: 31.25%
[  2/ 50]	FashionMNIST	Time 0.01869s	Training Accuracy: 45.80%	Test Accuracy: 46.88%
[  3/ 50]	       MNIST	Time 0.01086s	Training Accuracy: 40.43%	Test Accuracy: 28.12%
[  3/ 50]	FashionMNIST	Time 0.01481s	Training Accuracy: 58.50%	Test Accuracy: 59.38%
[  4/ 50]	       MNIST	Time 0.02361s	Training Accuracy: 51.17%	Test Accuracy: 43.75%
[  4/ 50]	FashionMNIST	Time 0.01156s	Training Accuracy: 63.87%	Test Accuracy: 59.38%
[  5/ 50]	       MNIST	Time 0.01288s	Training Accuracy: 55.47%	Test Accuracy: 40.62%
[  5/ 50]	FashionMNIST	Time 0.01239s	Training Accuracy: 70.51%	Test Accuracy: 62.50%
[  6/ 50]	       MNIST	Time 0.01306s	Training Accuracy: 62.60%	Test Accuracy: 37.50%
[  6/ 50]	FashionMNIST	Time 0.01437s	Training Accuracy: 75.78%	Test Accuracy: 59.38%
[  7/ 50]	       MNIST	Time 0.01222s	Training Accuracy: 70.90%	Test Accuracy: 43.75%
[  7/ 50]	FashionMNIST	Time 0.01356s	Training Accuracy: 75.59%	Test Accuracy: 65.62%
[  8/ 50]	       MNIST	Time 0.02179s	Training Accuracy: 76.17%	Test Accuracy: 46.88%
[  8/ 50]	FashionMNIST	Time 0.01508s	Training Accuracy: 81.05%	Test Accuracy: 65.62%
[  9/ 50]	       MNIST	Time 0.01268s	Training Accuracy: 80.08%	Test Accuracy: 46.88%
[  9/ 50]	FashionMNIST	Time 0.01144s	Training Accuracy: 84.18%	Test Accuracy: 65.62%
[ 10/ 50]	       MNIST	Time 0.01178s	Training Accuracy: 84.67%	Test Accuracy: 46.88%
[ 10/ 50]	FashionMNIST	Time 0.02060s	Training Accuracy: 88.28%	Test Accuracy: 62.50%
[ 11/ 50]	       MNIST	Time 0.01149s	Training Accuracy: 88.09%	Test Accuracy: 50.00%
[ 11/ 50]	FashionMNIST	Time 0.01253s	Training Accuracy: 89.94%	Test Accuracy: 68.75%
[ 12/ 50]	       MNIST	Time 0.01005s	Training Accuracy: 90.62%	Test Accuracy: 53.12%
[ 12/ 50]	FashionMNIST	Time 0.01802s	Training Accuracy: 91.21%	Test Accuracy: 68.75%
[ 13/ 50]	       MNIST	Time 0.02376s	Training Accuracy: 93.95%	Test Accuracy: 56.25%
[ 13/ 50]	FashionMNIST	Time 0.01477s	Training Accuracy: 93.16%	Test Accuracy: 71.88%
[ 14/ 50]	       MNIST	Time 0.01257s	Training Accuracy: 96.09%	Test Accuracy: 59.38%
[ 14/ 50]	FashionMNIST	Time 0.01167s	Training Accuracy: 95.41%	Test Accuracy: 68.75%
[ 15/ 50]	       MNIST	Time 0.01330s	Training Accuracy: 96.09%	Test Accuracy: 56.25%
[ 15/ 50]	FashionMNIST	Time 0.02210s	Training Accuracy: 95.12%	Test Accuracy: 68.75%
[ 16/ 50]	       MNIST	Time 0.01523s	Training Accuracy: 98.34%	Test Accuracy: 62.50%
[ 16/ 50]	FashionMNIST	Time 0.03077s	Training Accuracy: 96.00%	Test Accuracy: 71.88%
[ 17/ 50]	       MNIST	Time 0.01533s	Training Accuracy: 99.41%	Test Accuracy: 68.75%
[ 17/ 50]	FashionMNIST	Time 0.01221s	Training Accuracy: 97.75%	Test Accuracy: 75.00%
[ 18/ 50]	       MNIST	Time 0.01991s	Training Accuracy: 99.71%	Test Accuracy: 62.50%
[ 18/ 50]	FashionMNIST	Time 0.01631s	Training Accuracy: 97.46%	Test Accuracy: 71.88%
[ 19/ 50]	       MNIST	Time 0.01244s	Training Accuracy: 99.80%	Test Accuracy: 62.50%
[ 19/ 50]	FashionMNIST	Time 0.01137s	Training Accuracy: 99.02%	Test Accuracy: 75.00%
[ 20/ 50]	       MNIST	Time 0.01278s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 20/ 50]	FashionMNIST	Time 0.01156s	Training Accuracy: 99.02%	Test Accuracy: 71.88%
[ 21/ 50]	       MNIST	Time 0.01224s	Training Accuracy: 99.90%	Test Accuracy: 62.50%
[ 21/ 50]	FashionMNIST	Time 0.01007s	Training Accuracy: 99.02%	Test Accuracy: 71.88%
[ 22/ 50]	       MNIST	Time 0.02520s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 22/ 50]	FashionMNIST	Time 0.01954s	Training Accuracy: 99.61%	Test Accuracy: 71.88%
[ 23/ 50]	       MNIST	Time 0.01288s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 23/ 50]	FashionMNIST	Time 0.01047s	Training Accuracy: 99.80%	Test Accuracy: 68.75%
[ 24/ 50]	       MNIST	Time 0.01808s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 24/ 50]	FashionMNIST	Time 0.02442s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 25/ 50]	       MNIST	Time 0.01206s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 25/ 50]	FashionMNIST	Time 0.01277s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 26/ 50]	       MNIST	Time 0.01255s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 26/ 50]	FashionMNIST	Time 0.01302s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 27/ 50]	       MNIST	Time 0.02298s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 27/ 50]	FashionMNIST	Time 0.01316s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 28/ 50]	       MNIST	Time 0.01188s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 28/ 50]	FashionMNIST	Time 0.01447s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 29/ 50]	       MNIST	Time 0.01216s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 29/ 50]	FashionMNIST	Time 0.02139s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 30/ 50]	       MNIST	Time 0.01271s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 30/ 50]	FashionMNIST	Time 0.01503s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 31/ 50]	       MNIST	Time 0.01109s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 31/ 50]	FashionMNIST	Time 0.01308s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 32/ 50]	       MNIST	Time 0.04279s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 32/ 50]	FashionMNIST	Time 0.01339s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 33/ 50]	       MNIST	Time 0.01235s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 33/ 50]	FashionMNIST	Time 0.01298s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 34/ 50]	       MNIST	Time 0.01086s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 34/ 50]	FashionMNIST	Time 0.01590s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 35/ 50]	       MNIST	Time 0.01256s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 35/ 50]	FashionMNIST	Time 0.01273s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 36/ 50]	       MNIST	Time 0.02239s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 36/ 50]	FashionMNIST	Time 0.01289s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 37/ 50]	       MNIST	Time 0.01367s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 37/ 50]	FashionMNIST	Time 0.01291s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 38/ 50]	       MNIST	Time 0.01809s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 38/ 50]	FashionMNIST	Time 0.02153s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 39/ 50]	       MNIST	Time 0.01353s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 39/ 50]	FashionMNIST	Time 0.00949s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 40/ 50]	       MNIST	Time 0.01502s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 40/ 50]	FashionMNIST	Time 0.01310s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 41/ 50]	       MNIST	Time 0.02149s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 41/ 50]	FashionMNIST	Time 0.01387s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 42/ 50]	       MNIST	Time 0.01349s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 42/ 50]	FashionMNIST	Time 0.01121s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 43/ 50]	       MNIST	Time 0.01330s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 43/ 50]	FashionMNIST	Time 0.02169s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 44/ 50]	       MNIST	Time 0.01262s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 44/ 50]	FashionMNIST	Time 0.01229s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 45/ 50]	       MNIST	Time 0.01079s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 45/ 50]	FashionMNIST	Time 0.01320s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 46/ 50]	       MNIST	Time 0.02307s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 46/ 50]	FashionMNIST	Time 0.01592s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 47/ 50]	       MNIST	Time 0.01247s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 47/ 50]	FashionMNIST	Time 0.01271s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 48/ 50]	       MNIST	Time 0.01641s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 48/ 50]	FashionMNIST	Time 0.02638s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 49/ 50]	       MNIST	Time 0.01381s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 49/ 50]	FashionMNIST	Time 0.01177s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 50/ 50]	       MNIST	Time 0.02307s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 50/ 50]	FashionMNIST	Time 0.01272s	Training Accuracy: 100.00%	Test Accuracy: 71.88%

[FINAL]	       MNIST	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[FINAL]	FashionMNIST	Training Accuracy: 100.00%	Test Accuracy: 71.88%

```

## Appendix {#Appendix}

```julia
using InteractiveUtils
InteractiveUtils.versioninfo()

if @isdefined(MLDataDevices)
    if @isdefined(CUDA) && MLDataDevices.functional(CUDADevice)
        println()
        CUDA.versioninfo()
    end

    if @isdefined(AMDGPU) && MLDataDevices.functional(AMDGPUDevice)
        println()
        AMDGPU.versioninfo()
    end
end

```

```
Julia Version 1.12.7
Commit 6d172b025e4 (2026-08-15 08:05 UTC)
Build Info:
  Official https://julialang.org release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 4 × AMD EPYC 9V74 80-Core Processor
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, znver4)
  GC: Built with stock GC
Threads: 4 default, 1 interactive, 4 GC (on 4 virtual cores)
Environment:
  JULIA_DEBUG = Literate
  LD_LIBRARY_PATH = 
  JULIA_NUM_THREADS = 4
  JULIA_CPU_HARD_MEMORY_LIMIT = 100%
  JULIA_PKG_PRECOMPILE_AUTO = 0

```

***

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
