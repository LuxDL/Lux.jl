---
url: /previews/PR1677/tutorials/intermediate/3_HyperNet.md
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
[  1/ 50]	       MNIST	Time 49.82264s	Training Accuracy: 34.57%	Test Accuracy: 37.50%
[  1/ 50]	FashionMNIST	Time 0.09683s	Training Accuracy: 32.62%	Test Accuracy: 43.75%
[  2/ 50]	       MNIST	Time 0.09230s	Training Accuracy: 36.23%	Test Accuracy: 31.25%
[  2/ 50]	FashionMNIST	Time 0.09068s	Training Accuracy: 46.19%	Test Accuracy: 46.88%
[  3/ 50]	       MNIST	Time 0.08821s	Training Accuracy: 40.23%	Test Accuracy: 31.25%
[  3/ 50]	FashionMNIST	Time 0.09175s	Training Accuracy: 53.52%	Test Accuracy: 62.50%
[  4/ 50]	       MNIST	Time 0.09112s	Training Accuracy: 53.32%	Test Accuracy: 43.75%
[  4/ 50]	FashionMNIST	Time 0.10237s	Training Accuracy: 62.11%	Test Accuracy: 62.50%
[  5/ 50]	       MNIST	Time 0.09049s	Training Accuracy: 56.74%	Test Accuracy: 37.50%
[  5/ 50]	FashionMNIST	Time 0.09057s	Training Accuracy: 67.97%	Test Accuracy: 62.50%
[  6/ 50]	       MNIST	Time 0.08811s	Training Accuracy: 63.28%	Test Accuracy: 40.62%
[  6/ 50]	FashionMNIST	Time 0.09091s	Training Accuracy: 74.32%	Test Accuracy: 56.25%
[  7/ 50]	       MNIST	Time 0.09010s	Training Accuracy: 68.95%	Test Accuracy: 46.88%
[  7/ 50]	FashionMNIST	Time 0.08902s	Training Accuracy: 75.20%	Test Accuracy: 56.25%
[  8/ 50]	       MNIST	Time 0.09076s	Training Accuracy: 76.27%	Test Accuracy: 46.88%
[  8/ 50]	FashionMNIST	Time 0.09078s	Training Accuracy: 79.98%	Test Accuracy: 65.62%
[  9/ 50]	       MNIST	Time 0.09032s	Training Accuracy: 80.86%	Test Accuracy: 53.12%
[  9/ 50]	FashionMNIST	Time 0.09474s	Training Accuracy: 84.47%	Test Accuracy: 62.50%
[ 10/ 50]	       MNIST	Time 0.08981s	Training Accuracy: 82.13%	Test Accuracy: 46.88%
[ 10/ 50]	FashionMNIST	Time 0.08850s	Training Accuracy: 88.09%	Test Accuracy: 65.62%
[ 11/ 50]	       MNIST	Time 0.09803s	Training Accuracy: 88.48%	Test Accuracy: 50.00%
[ 11/ 50]	FashionMNIST	Time 0.08988s	Training Accuracy: 89.36%	Test Accuracy: 62.50%
[ 12/ 50]	       MNIST	Time 0.09005s	Training Accuracy: 89.65%	Test Accuracy: 53.12%
[ 12/ 50]	FashionMNIST	Time 0.09701s	Training Accuracy: 91.02%	Test Accuracy: 65.62%
[ 13/ 50]	       MNIST	Time 0.08860s	Training Accuracy: 93.07%	Test Accuracy: 53.12%
[ 13/ 50]	FashionMNIST	Time 0.08857s	Training Accuracy: 93.75%	Test Accuracy: 71.88%
[ 14/ 50]	       MNIST	Time 0.09875s	Training Accuracy: 93.85%	Test Accuracy: 53.12%
[ 14/ 50]	FashionMNIST	Time 0.08927s	Training Accuracy: 94.53%	Test Accuracy: 68.75%
[ 15/ 50]	       MNIST	Time 0.08932s	Training Accuracy: 96.09%	Test Accuracy: 56.25%
[ 15/ 50]	FashionMNIST	Time 0.09720s	Training Accuracy: 93.55%	Test Accuracy: 68.75%
[ 16/ 50]	       MNIST	Time 0.09131s	Training Accuracy: 98.24%	Test Accuracy: 62.50%
[ 16/ 50]	FashionMNIST	Time 0.09116s	Training Accuracy: 97.17%	Test Accuracy: 71.88%
[ 17/ 50]	       MNIST	Time 0.09856s	Training Accuracy: 99.22%	Test Accuracy: 56.25%
[ 17/ 50]	FashionMNIST	Time 0.08946s	Training Accuracy: 97.66%	Test Accuracy: 75.00%
[ 18/ 50]	       MNIST	Time 0.08901s	Training Accuracy: 99.71%	Test Accuracy: 59.38%
[ 18/ 50]	FashionMNIST	Time 0.09653s	Training Accuracy: 97.27%	Test Accuracy: 68.75%
[ 19/ 50]	       MNIST	Time 0.08969s	Training Accuracy: 99.80%	Test Accuracy: 59.38%
[ 19/ 50]	FashionMNIST	Time 0.08837s	Training Accuracy: 98.63%	Test Accuracy: 75.00%
[ 20/ 50]	       MNIST	Time 0.09223s	Training Accuracy: 99.90%	Test Accuracy: 56.25%
[ 20/ 50]	FashionMNIST	Time 0.08841s	Training Accuracy: 99.02%	Test Accuracy: 75.00%
[ 21/ 50]	       MNIST	Time 0.08950s	Training Accuracy: 99.90%	Test Accuracy: 62.50%
[ 21/ 50]	FashionMNIST	Time 0.08909s	Training Accuracy: 99.02%	Test Accuracy: 71.88%
[ 22/ 50]	       MNIST	Time 0.08969s	Training Accuracy: 99.90%	Test Accuracy: 59.38%
[ 22/ 50]	FashionMNIST	Time 0.08896s	Training Accuracy: 99.41%	Test Accuracy: 75.00%
[ 23/ 50]	       MNIST	Time 0.09185s	Training Accuracy: 99.90%	Test Accuracy: 59.38%
[ 23/ 50]	FashionMNIST	Time 0.08775s	Training Accuracy: 99.71%	Test Accuracy: 71.88%
[ 24/ 50]	       MNIST	Time 0.08896s	Training Accuracy: 99.90%	Test Accuracy: 59.38%
[ 24/ 50]	FashionMNIST	Time 0.09001s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 25/ 50]	       MNIST	Time 0.09046s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 25/ 50]	FashionMNIST	Time 0.09088s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 26/ 50]	       MNIST	Time 0.08845s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 26/ 50]	FashionMNIST	Time 0.08751s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 27/ 50]	       MNIST	Time 0.08846s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 27/ 50]	FashionMNIST	Time 0.08941s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 28/ 50]	       MNIST	Time 0.09094s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 28/ 50]	FashionMNIST	Time 0.09039s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 29/ 50]	       MNIST	Time 0.09003s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 29/ 50]	FashionMNIST	Time 0.08961s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 30/ 50]	       MNIST	Time 0.10365s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 30/ 50]	FashionMNIST	Time 0.08997s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 31/ 50]	       MNIST	Time 0.09051s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 31/ 50]	FashionMNIST	Time 0.09730s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 32/ 50]	       MNIST	Time 0.08957s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 32/ 50]	FashionMNIST	Time 0.08826s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 33/ 50]	       MNIST	Time 0.09604s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 33/ 50]	FashionMNIST	Time 0.09105s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 34/ 50]	       MNIST	Time 0.08904s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 34/ 50]	FashionMNIST	Time 0.09868s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 35/ 50]	       MNIST	Time 0.09070s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 35/ 50]	FashionMNIST	Time 0.08769s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 36/ 50]	       MNIST	Time 0.09594s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 36/ 50]	FashionMNIST	Time 0.08841s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 37/ 50]	       MNIST	Time 0.08801s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 37/ 50]	FashionMNIST	Time 0.09658s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 38/ 50]	       MNIST	Time 0.09084s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 38/ 50]	FashionMNIST	Time 0.09057s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 39/ 50]	       MNIST	Time 0.08995s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 39/ 50]	FashionMNIST	Time 0.08768s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 40/ 50]	       MNIST	Time 0.08797s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 40/ 50]	FashionMNIST	Time 0.09037s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 41/ 50]	       MNIST	Time 0.08855s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 41/ 50]	FashionMNIST	Time 0.08821s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 42/ 50]	       MNIST	Time 0.08885s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 42/ 50]	FashionMNIST	Time 0.08933s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 43/ 50]	       MNIST	Time 0.08962s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 43/ 50]	FashionMNIST	Time 0.08872s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 44/ 50]	       MNIST	Time 0.09213s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 44/ 50]	FashionMNIST	Time 0.08934s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 45/ 50]	       MNIST	Time 0.08967s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 45/ 50]	FashionMNIST	Time 0.08781s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 46/ 50]	       MNIST	Time 0.09155s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 46/ 50]	FashionMNIST	Time 0.09048s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 47/ 50]	       MNIST	Time 0.09118s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 47/ 50]	FashionMNIST	Time 0.08758s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 48/ 50]	       MNIST	Time 0.08786s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 48/ 50]	FashionMNIST	Time 0.08896s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 49/ 50]	       MNIST	Time 0.09790s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 49/ 50]	FashionMNIST	Time 0.09115s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 50/ 50]	       MNIST	Time 0.08924s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 50/ 50]	FashionMNIST	Time 0.09951s	Training Accuracy: 100.00%	Test Accuracy: 71.88%

[FINAL]	       MNIST	Training Accuracy: 100.00%	Test Accuracy: 59.38%
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
Julia Version 1.12.5
Commit 5fe89b8ddc1 (2026-02-09 16:05 UTC)
Build Info:
  Official https://julialang.org release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 4 × AMD EPYC 7763 64-Core Processor
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, znver3)
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
