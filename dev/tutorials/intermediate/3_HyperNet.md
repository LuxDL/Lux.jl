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
[  1/ 50]	       MNIST	Time 38.39190s	Training Accuracy: 34.57%	Test Accuracy: 37.50%
[  1/ 50]	FashionMNIST	Time 0.07713s	Training Accuracy: 32.62%	Test Accuracy: 43.75%
[  2/ 50]	       MNIST	Time 0.08074s	Training Accuracy: 36.72%	Test Accuracy: 34.38%
[  2/ 50]	FashionMNIST	Time 0.09426s	Training Accuracy: 45.61%	Test Accuracy: 50.00%
[  3/ 50]	       MNIST	Time 0.07720s	Training Accuracy: 41.02%	Test Accuracy: 28.12%
[  3/ 50]	FashionMNIST	Time 0.07486s	Training Accuracy: 57.13%	Test Accuracy: 59.38%
[  4/ 50]	       MNIST	Time 0.08937s	Training Accuracy: 51.86%	Test Accuracy: 40.62%
[  4/ 50]	FashionMNIST	Time 0.07422s	Training Accuracy: 64.36%	Test Accuracy: 56.25%
[  5/ 50]	       MNIST	Time 0.07552s	Training Accuracy: 58.30%	Test Accuracy: 37.50%
[  5/ 50]	FashionMNIST	Time 0.07509s	Training Accuracy: 69.14%	Test Accuracy: 56.25%
[  6/ 50]	       MNIST	Time 0.07305s	Training Accuracy: 64.55%	Test Accuracy: 34.38%
[  6/ 50]	FashionMNIST	Time 0.07526s	Training Accuracy: 74.90%	Test Accuracy: 53.12%
[  7/ 50]	       MNIST	Time 0.07319s	Training Accuracy: 70.12%	Test Accuracy: 34.38%
[  7/ 50]	FashionMNIST	Time 0.07572s	Training Accuracy: 76.17%	Test Accuracy: 53.12%
[  8/ 50]	       MNIST	Time 0.07414s	Training Accuracy: 75.88%	Test Accuracy: 43.75%
[  8/ 50]	FashionMNIST	Time 0.07938s	Training Accuracy: 80.96%	Test Accuracy: 65.62%
[  9/ 50]	       MNIST	Time 0.07497s	Training Accuracy: 79.98%	Test Accuracy: 43.75%
[  9/ 50]	FashionMNIST	Time 0.07740s	Training Accuracy: 82.52%	Test Accuracy: 62.50%
[ 10/ 50]	       MNIST	Time 0.07566s	Training Accuracy: 85.35%	Test Accuracy: 53.12%
[ 10/ 50]	FashionMNIST	Time 0.08570s	Training Accuracy: 88.18%	Test Accuracy: 56.25%
[ 11/ 50]	       MNIST	Time 0.07629s	Training Accuracy: 88.96%	Test Accuracy: 53.12%
[ 11/ 50]	FashionMNIST	Time 0.07475s	Training Accuracy: 90.14%	Test Accuracy: 65.62%
[ 12/ 50]	       MNIST	Time 0.08558s	Training Accuracy: 91.50%	Test Accuracy: 50.00%
[ 12/ 50]	FashionMNIST	Time 0.07717s	Training Accuracy: 91.21%	Test Accuracy: 65.62%
[ 13/ 50]	       MNIST	Time 0.07654s	Training Accuracy: 93.75%	Test Accuracy: 56.25%
[ 13/ 50]	FashionMNIST	Time 0.08434s	Training Accuracy: 92.48%	Test Accuracy: 65.62%
[ 14/ 50]	       MNIST	Time 0.07657s	Training Accuracy: 95.90%	Test Accuracy: 53.12%
[ 14/ 50]	FashionMNIST	Time 0.07618s	Training Accuracy: 95.12%	Test Accuracy: 68.75%
[ 15/ 50]	       MNIST	Time 0.08415s	Training Accuracy: 96.19%	Test Accuracy: 53.12%
[ 15/ 50]	FashionMNIST	Time 0.07433s	Training Accuracy: 93.46%	Test Accuracy: 68.75%
[ 16/ 50]	       MNIST	Time 0.07596s	Training Accuracy: 98.73%	Test Accuracy: 62.50%
[ 16/ 50]	FashionMNIST	Time 0.07331s	Training Accuracy: 96.58%	Test Accuracy: 68.75%
[ 17/ 50]	       MNIST	Time 0.07478s	Training Accuracy: 99.41%	Test Accuracy: 56.25%
[ 17/ 50]	FashionMNIST	Time 0.07453s	Training Accuracy: 97.66%	Test Accuracy: 71.88%
[ 18/ 50]	       MNIST	Time 0.07644s	Training Accuracy: 99.51%	Test Accuracy: 59.38%
[ 18/ 50]	FashionMNIST	Time 0.07578s	Training Accuracy: 97.56%	Test Accuracy: 65.62%
[ 19/ 50]	       MNIST	Time 0.07765s	Training Accuracy: 99.80%	Test Accuracy: 53.12%
[ 19/ 50]	FashionMNIST	Time 0.07812s	Training Accuracy: 99.22%	Test Accuracy: 68.75%
[ 20/ 50]	       MNIST	Time 0.07923s	Training Accuracy: 99.90%	Test Accuracy: 53.12%
[ 20/ 50]	FashionMNIST	Time 0.07933s	Training Accuracy: 98.83%	Test Accuracy: 68.75%
[ 21/ 50]	       MNIST	Time 0.07980s	Training Accuracy: 99.90%	Test Accuracy: 53.12%
[ 21/ 50]	FashionMNIST	Time 0.07985s	Training Accuracy: 99.02%	Test Accuracy: 68.75%
[ 22/ 50]	       MNIST	Time 0.07888s	Training Accuracy: 99.90%	Test Accuracy: 62.50%
[ 22/ 50]	FashionMNIST	Time 0.07776s	Training Accuracy: 99.51%	Test Accuracy: 68.75%
[ 23/ 50]	       MNIST	Time 0.08847s	Training Accuracy: 99.90%	Test Accuracy: 56.25%
[ 23/ 50]	FashionMNIST	Time 0.07658s	Training Accuracy: 99.80%	Test Accuracy: 65.62%
[ 24/ 50]	       MNIST	Time 0.07604s	Training Accuracy: 99.90%	Test Accuracy: 59.38%
[ 24/ 50]	FashionMNIST	Time 0.08668s	Training Accuracy: 99.51%	Test Accuracy: 65.62%
[ 25/ 50]	       MNIST	Time 0.07745s	Training Accuracy: 99.90%	Test Accuracy: 59.38%
[ 25/ 50]	FashionMNIST	Time 0.07741s	Training Accuracy: 99.80%	Test Accuracy: 68.75%
[ 26/ 50]	       MNIST	Time 0.08681s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 26/ 50]	FashionMNIST	Time 0.07933s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 27/ 50]	       MNIST	Time 0.07726s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 27/ 50]	FashionMNIST	Time 0.07715s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 28/ 50]	       MNIST	Time 0.07637s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 28/ 50]	FashionMNIST	Time 0.07869s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 29/ 50]	       MNIST	Time 0.07736s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 29/ 50]	FashionMNIST	Time 0.07818s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 30/ 50]	       MNIST	Time 0.07736s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 30/ 50]	FashionMNIST	Time 0.07636s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 31/ 50]	       MNIST	Time 0.07758s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 31/ 50]	FashionMNIST	Time 0.07770s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 32/ 50]	       MNIST	Time 0.07799s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 32/ 50]	FashionMNIST	Time 0.08506s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 33/ 50]	       MNIST	Time 0.07954s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 33/ 50]	FashionMNIST	Time 0.07915s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 34/ 50]	       MNIST	Time 0.09106s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 34/ 50]	FashionMNIST	Time 0.07713s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 35/ 50]	       MNIST	Time 0.07797s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 35/ 50]	FashionMNIST	Time 0.08719s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 36/ 50]	       MNIST	Time 0.07823s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 36/ 50]	FashionMNIST	Time 0.07809s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 37/ 50]	       MNIST	Time 0.08702s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 37/ 50]	FashionMNIST	Time 0.07608s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 38/ 50]	       MNIST	Time 0.07823s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 38/ 50]	FashionMNIST	Time 0.07628s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 39/ 50]	       MNIST	Time 0.07741s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 39/ 50]	FashionMNIST	Time 0.07714s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 40/ 50]	       MNIST	Time 0.07627s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 40/ 50]	FashionMNIST	Time 0.07701s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 41/ 50]	       MNIST	Time 0.07594s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 41/ 50]	FashionMNIST	Time 0.07522s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 42/ 50]	       MNIST	Time 0.07559s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 42/ 50]	FashionMNIST	Time 0.07601s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 43/ 50]	       MNIST	Time 0.07651s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 43/ 50]	FashionMNIST	Time 0.07687s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 44/ 50]	       MNIST	Time 0.07532s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 44/ 50]	FashionMNIST	Time 0.07498s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 45/ 50]	       MNIST	Time 0.08511s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 45/ 50]	FashionMNIST	Time 0.07584s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 46/ 50]	       MNIST	Time 0.07500s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 46/ 50]	FashionMNIST	Time 0.08579s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 47/ 50]	       MNIST	Time 0.07654s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 47/ 50]	FashionMNIST	Time 0.07553s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 48/ 50]	       MNIST	Time 0.08506s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 48/ 50]	FashionMNIST	Time 0.07572s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 49/ 50]	       MNIST	Time 0.07517s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 49/ 50]	FashionMNIST	Time 0.08403s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 50/ 50]	       MNIST	Time 0.07522s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 50/ 50]	FashionMNIST	Time 0.07619s	Training Accuracy: 100.00%	Test Accuracy: 65.62%

[FINAL]	       MNIST	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[FINAL]	FashionMNIST	Training Accuracy: 100.00%	Test Accuracy: 65.62%

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
