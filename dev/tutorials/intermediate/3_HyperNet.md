---
url: /dev/tutorials/intermediate/3_HyperNet.md
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
[  1/ 50]	       MNIST	Time 48.95168s	Training Accuracy: 34.57%	Test Accuracy: 37.50%
[  1/ 50]	FashionMNIST	Time 0.09104s	Training Accuracy: 32.52%	Test Accuracy: 43.75%
[  2/ 50]	       MNIST	Time 0.09154s	Training Accuracy: 36.33%	Test Accuracy: 34.38%
[  2/ 50]	FashionMNIST	Time 0.09255s	Training Accuracy: 46.19%	Test Accuracy: 46.88%
[  3/ 50]	       MNIST	Time 0.08873s	Training Accuracy: 42.68%	Test Accuracy: 28.12%
[  3/ 50]	FashionMNIST	Time 0.08888s	Training Accuracy: 56.64%	Test Accuracy: 56.25%
[  4/ 50]	       MNIST	Time 0.08997s	Training Accuracy: 51.37%	Test Accuracy: 37.50%
[  4/ 50]	FashionMNIST	Time 0.09015s	Training Accuracy: 65.23%	Test Accuracy: 62.50%
[  5/ 50]	       MNIST	Time 0.08882s	Training Accuracy: 56.64%	Test Accuracy: 40.62%
[  5/ 50]	FashionMNIST	Time 0.09817s	Training Accuracy: 70.51%	Test Accuracy: 59.38%
[  6/ 50]	       MNIST	Time 0.08931s	Training Accuracy: 61.62%	Test Accuracy: 37.50%
[  6/ 50]	FashionMNIST	Time 0.08912s	Training Accuracy: 75.78%	Test Accuracy: 56.25%
[  7/ 50]	       MNIST	Time 0.09753s	Training Accuracy: 67.77%	Test Accuracy: 43.75%
[  7/ 50]	FashionMNIST	Time 0.08926s	Training Accuracy: 75.59%	Test Accuracy: 62.50%
[  8/ 50]	       MNIST	Time 0.08951s	Training Accuracy: 74.90%	Test Accuracy: 46.88%
[  8/ 50]	FashionMNIST	Time 0.09832s	Training Accuracy: 80.96%	Test Accuracy: 62.50%
[  9/ 50]	       MNIST	Time 0.08859s	Training Accuracy: 81.05%	Test Accuracy: 53.12%
[  9/ 50]	FashionMNIST	Time 0.08891s	Training Accuracy: 84.77%	Test Accuracy: 62.50%
[ 10/ 50]	       MNIST	Time 0.09833s	Training Accuracy: 82.52%	Test Accuracy: 53.12%
[ 10/ 50]	FashionMNIST	Time 0.08924s	Training Accuracy: 88.18%	Test Accuracy: 59.38%
[ 11/ 50]	       MNIST	Time 0.08937s	Training Accuracy: 86.43%	Test Accuracy: 53.12%
[ 11/ 50]	FashionMNIST	Time 0.08976s	Training Accuracy: 89.84%	Test Accuracy: 62.50%
[ 12/ 50]	       MNIST	Time 0.09095s	Training Accuracy: 90.04%	Test Accuracy: 50.00%
[ 12/ 50]	FashionMNIST	Time 0.08966s	Training Accuracy: 92.09%	Test Accuracy: 68.75%
[ 13/ 50]	       MNIST	Time 0.09800s	Training Accuracy: 94.43%	Test Accuracy: 59.38%
[ 13/ 50]	FashionMNIST	Time 0.08975s	Training Accuracy: 93.46%	Test Accuracy: 68.75%
[ 14/ 50]	       MNIST	Time 0.09085s	Training Accuracy: 94.53%	Test Accuracy: 56.25%
[ 14/ 50]	FashionMNIST	Time 0.08913s	Training Accuracy: 95.51%	Test Accuracy: 68.75%
[ 15/ 50]	       MNIST	Time 0.08859s	Training Accuracy: 96.29%	Test Accuracy: 71.88%
[ 15/ 50]	FashionMNIST	Time 0.08901s	Training Accuracy: 94.24%	Test Accuracy: 68.75%
[ 16/ 50]	       MNIST	Time 0.08977s	Training Accuracy: 98.63%	Test Accuracy: 59.38%
[ 16/ 50]	FashionMNIST	Time 0.08905s	Training Accuracy: 95.51%	Test Accuracy: 71.88%
[ 17/ 50]	       MNIST	Time 0.09004s	Training Accuracy: 99.32%	Test Accuracy: 59.38%
[ 17/ 50]	FashionMNIST	Time 0.08931s	Training Accuracy: 97.36%	Test Accuracy: 71.88%
[ 18/ 50]	       MNIST	Time 0.08937s	Training Accuracy: 99.61%	Test Accuracy: 71.88%
[ 18/ 50]	FashionMNIST	Time 0.09765s	Training Accuracy: 97.46%	Test Accuracy: 71.88%
[ 19/ 50]	       MNIST	Time 0.09008s	Training Accuracy: 99.61%	Test Accuracy: 65.62%
[ 19/ 50]	FashionMNIST	Time 0.08942s	Training Accuracy: 98.63%	Test Accuracy: 65.62%
[ 20/ 50]	       MNIST	Time 0.09765s	Training Accuracy: 99.90%	Test Accuracy: 59.38%
[ 20/ 50]	FashionMNIST	Time 0.08879s	Training Accuracy: 98.63%	Test Accuracy: 71.88%
[ 21/ 50]	       MNIST	Time 0.08911s	Training Accuracy: 99.90%	Test Accuracy: 65.62%
[ 21/ 50]	FashionMNIST	Time 0.09814s	Training Accuracy: 98.93%	Test Accuracy: 71.88%
[ 22/ 50]	       MNIST	Time 0.09128s	Training Accuracy: 99.90%	Test Accuracy: 68.75%
[ 22/ 50]	FashionMNIST	Time 0.08952s	Training Accuracy: 99.51%	Test Accuracy: 71.88%
[ 23/ 50]	       MNIST	Time 0.09756s	Training Accuracy: 99.90%	Test Accuracy: 65.62%
[ 23/ 50]	FashionMNIST	Time 0.08892s	Training Accuracy: 99.71%	Test Accuracy: 68.75%
[ 24/ 50]	       MNIST	Time 0.08825s	Training Accuracy: 99.90%	Test Accuracy: 65.62%
[ 24/ 50]	FashionMNIST	Time 0.08943s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 25/ 50]	       MNIST	Time 0.08889s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 25/ 50]	FashionMNIST	Time 0.08869s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 26/ 50]	       MNIST	Time 0.08916s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 26/ 50]	FashionMNIST	Time 0.08870s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 27/ 50]	       MNIST	Time 0.08861s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 27/ 50]	FashionMNIST	Time 0.08974s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 28/ 50]	       MNIST	Time 0.08857s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 28/ 50]	FashionMNIST	Time 0.08910s	Training Accuracy: 99.90%	Test Accuracy: 68.75%
[ 29/ 50]	       MNIST	Time 0.09037s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 29/ 50]	FashionMNIST	Time 0.08979s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 30/ 50]	       MNIST	Time 0.09810s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 30/ 50]	FashionMNIST	Time 0.09035s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 31/ 50]	       MNIST	Time 0.08924s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 31/ 50]	FashionMNIST	Time 0.09852s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 32/ 50]	       MNIST	Time 0.08928s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 32/ 50]	FashionMNIST	Time 0.08886s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 33/ 50]	       MNIST	Time 0.09906s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 33/ 50]	FashionMNIST	Time 0.08975s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 34/ 50]	       MNIST	Time 0.09844s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 34/ 50]	FashionMNIST	Time 0.09775s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 35/ 50]	       MNIST	Time 0.08903s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 35/ 50]	FashionMNIST	Time 0.08901s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 36/ 50]	       MNIST	Time 0.09778s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 36/ 50]	FashionMNIST	Time 0.08863s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 37/ 50]	       MNIST	Time 0.08853s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 37/ 50]	FashionMNIST	Time 0.08924s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 38/ 50]	       MNIST	Time 0.08953s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 38/ 50]	FashionMNIST	Time 0.08908s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 39/ 50]	       MNIST	Time 0.09063s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 39/ 50]	FashionMNIST	Time 0.08909s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 40/ 50]	       MNIST	Time 0.08969s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 40/ 50]	FashionMNIST	Time 0.09050s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 41/ 50]	       MNIST	Time 0.09187s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 41/ 50]	FashionMNIST	Time 0.08904s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 42/ 50]	       MNIST	Time 0.08978s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 42/ 50]	FashionMNIST	Time 0.08939s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 43/ 50]	       MNIST	Time 0.10038s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 43/ 50]	FashionMNIST	Time 0.09049s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 44/ 50]	       MNIST	Time 0.10011s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 44/ 50]	FashionMNIST	Time 0.10776s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 45/ 50]	       MNIST	Time 0.09068s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 45/ 50]	FashionMNIST	Time 0.09022s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 46/ 50]	       MNIST	Time 0.09803s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 46/ 50]	FashionMNIST	Time 0.08875s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 47/ 50]	       MNIST	Time 0.09093s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 47/ 50]	FashionMNIST	Time 0.09692s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 48/ 50]	       MNIST	Time 0.08884s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 48/ 50]	FashionMNIST	Time 0.08904s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 49/ 50]	       MNIST	Time 0.09042s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 49/ 50]	FashionMNIST	Time 0.08861s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 50/ 50]	       MNIST	Time 0.09052s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 50/ 50]	FashionMNIST	Time 0.09149s	Training Accuracy: 100.00%	Test Accuracy: 68.75%

[FINAL]	       MNIST	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[FINAL]	FashionMNIST	Training Accuracy: 100.00%	Test Accuracy: 68.75%

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
