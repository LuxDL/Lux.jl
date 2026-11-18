---
url: /previews/PR1818/tutorials/intermediate/3_HyperNet.md
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
[  1/ 50]	       MNIST	Time 37.95016s	Training Accuracy: 34.28%	Test Accuracy: 37.50%
[  1/ 50]	FashionMNIST	Time 0.01766s	Training Accuracy: 36.62%	Test Accuracy: 43.75%
[  2/ 50]	       MNIST	Time 0.05951s	Training Accuracy: 45.51%	Test Accuracy: 40.62%
[  2/ 50]	FashionMNIST	Time 0.01426s	Training Accuracy: 59.08%	Test Accuracy: 56.25%
[  3/ 50]	       MNIST	Time 0.01471s	Training Accuracy: 59.57%	Test Accuracy: 46.88%
[  3/ 50]	FashionMNIST	Time 0.03533s	Training Accuracy: 69.63%	Test Accuracy: 62.50%
[  4/ 50]	       MNIST	Time 0.01423s	Training Accuracy: 69.43%	Test Accuracy: 56.25%
[  4/ 50]	FashionMNIST	Time 0.01554s	Training Accuracy: 74.51%	Test Accuracy: 68.75%
[  5/ 50]	       MNIST	Time 0.02424s	Training Accuracy: 76.37%	Test Accuracy: 59.38%
[  5/ 50]	FashionMNIST	Time 0.01700s	Training Accuracy: 78.61%	Test Accuracy: 62.50%
[  6/ 50]	       MNIST	Time 0.01637s	Training Accuracy: 82.32%	Test Accuracy: 56.25%
[  6/ 50]	FashionMNIST	Time 0.01573s	Training Accuracy: 81.54%	Test Accuracy: 68.75%
[  7/ 50]	       MNIST	Time 0.02019s	Training Accuracy: 87.89%	Test Accuracy: 56.25%
[  7/ 50]	FashionMNIST	Time 0.02408s	Training Accuracy: 83.11%	Test Accuracy: 65.62%
[  8/ 50]	       MNIST	Time 0.01780s	Training Accuracy: 90.92%	Test Accuracy: 59.38%
[  8/ 50]	FashionMNIST	Time 0.01527s	Training Accuracy: 88.77%	Test Accuracy: 71.88%
[  9/ 50]	       MNIST	Time 0.01538s	Training Accuracy: 91.02%	Test Accuracy: 59.38%
[  9/ 50]	FashionMNIST	Time 0.01772s	Training Accuracy: 91.41%	Test Accuracy: 71.88%
[ 10/ 50]	       MNIST	Time 0.02659s	Training Accuracy: 96.29%	Test Accuracy: 62.50%
[ 10/ 50]	FashionMNIST	Time 0.01481s	Training Accuracy: 94.34%	Test Accuracy: 71.88%
[ 11/ 50]	       MNIST	Time 0.01646s	Training Accuracy: 97.75%	Test Accuracy: 62.50%
[ 11/ 50]	FashionMNIST	Time 0.01848s	Training Accuracy: 96.19%	Test Accuracy: 75.00%
[ 12/ 50]	       MNIST	Time 0.01635s	Training Accuracy: 99.12%	Test Accuracy: 62.50%
[ 12/ 50]	FashionMNIST	Time 0.02645s	Training Accuracy: 97.07%	Test Accuracy: 78.12%
[ 13/ 50]	       MNIST	Time 0.01520s	Training Accuracy: 99.22%	Test Accuracy: 56.25%
[ 13/ 50]	FashionMNIST	Time 0.01535s	Training Accuracy: 98.05%	Test Accuracy: 71.88%
[ 14/ 50]	       MNIST	Time 0.01678s	Training Accuracy: 99.80%	Test Accuracy: 59.38%
[ 14/ 50]	FashionMNIST	Time 0.01578s	Training Accuracy: 97.27%	Test Accuracy: 75.00%
[ 15/ 50]	       MNIST	Time 0.02600s	Training Accuracy: 99.90%	Test Accuracy: 62.50%
[ 15/ 50]	FashionMNIST	Time 0.01752s	Training Accuracy: 98.05%	Test Accuracy: 71.88%
[ 16/ 50]	       MNIST	Time 0.01553s	Training Accuracy: 100.00%	Test Accuracy: 56.25%
[ 16/ 50]	FashionMNIST	Time 0.02339s	Training Accuracy: 98.73%	Test Accuracy: 78.12%
[ 17/ 50]	       MNIST	Time 0.01566s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 17/ 50]	FashionMNIST	Time 0.01711s	Training Accuracy: 98.34%	Test Accuracy: 78.12%
[ 18/ 50]	       MNIST	Time 0.01771s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 18/ 50]	FashionMNIST	Time 0.01722s	Training Accuracy: 99.71%	Test Accuracy: 75.00%
[ 19/ 50]	       MNIST	Time 0.02343s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 19/ 50]	FashionMNIST	Time 0.01805s	Training Accuracy: 100.00%	Test Accuracy: 81.25%
[ 20/ 50]	       MNIST	Time 0.01384s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 20/ 50]	FashionMNIST	Time 0.01656s	Training Accuracy: 99.51%	Test Accuracy: 75.00%
[ 21/ 50]	       MNIST	Time 0.01395s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 21/ 50]	FashionMNIST	Time 0.02196s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 22/ 50]	       MNIST	Time 0.01478s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 22/ 50]	FashionMNIST	Time 0.01466s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 23/ 50]	       MNIST	Time 0.01589s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 23/ 50]	FashionMNIST	Time 0.01426s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 24/ 50]	       MNIST	Time 0.02486s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 24/ 50]	FashionMNIST	Time 0.01849s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 25/ 50]	       MNIST	Time 0.01512s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 25/ 50]	FashionMNIST	Time 0.01816s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 26/ 50]	       MNIST	Time 0.01358s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 26/ 50]	FashionMNIST	Time 0.02420s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 27/ 50]	       MNIST	Time 0.01527s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 27/ 50]	FashionMNIST	Time 0.01538s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 28/ 50]	       MNIST	Time 0.01354s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 28/ 50]	FashionMNIST	Time 0.01455s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 29/ 50]	       MNIST	Time 0.02162s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 29/ 50]	FashionMNIST	Time 0.01738s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 30/ 50]	       MNIST	Time 0.01513s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 30/ 50]	FashionMNIST	Time 0.01767s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 31/ 50]	       MNIST	Time 0.01711s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 31/ 50]	FashionMNIST	Time 0.01690s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 32/ 50]	       MNIST	Time 0.01738s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 32/ 50]	FashionMNIST	Time 0.01751s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 33/ 50]	       MNIST	Time 0.02564s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 33/ 50]	FashionMNIST	Time 0.01559s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 34/ 50]	       MNIST	Time 0.01686s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 34/ 50]	FashionMNIST	Time 0.01490s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 35/ 50]	       MNIST	Time 0.01532s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 35/ 50]	FashionMNIST	Time 0.02481s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 36/ 50]	       MNIST	Time 0.01632s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 36/ 50]	FashionMNIST	Time 0.01637s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 37/ 50]	       MNIST	Time 0.01457s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 37/ 50]	FashionMNIST	Time 0.01710s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 38/ 50]	       MNIST	Time 0.02468s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 38/ 50]	FashionMNIST	Time 0.01825s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 39/ 50]	       MNIST	Time 0.01728s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 39/ 50]	FashionMNIST	Time 0.01539s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 40/ 50]	       MNIST	Time 0.01873s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 40/ 50]	FashionMNIST	Time 0.05234s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 41/ 50]	       MNIST	Time 0.01599s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 41/ 50]	FashionMNIST	Time 0.01754s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 42/ 50]	       MNIST	Time 0.01470s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 42/ 50]	FashionMNIST	Time 0.01851s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 43/ 50]	       MNIST	Time 0.02382s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 43/ 50]	FashionMNIST	Time 0.01844s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 44/ 50]	       MNIST	Time 0.01554s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 44/ 50]	FashionMNIST	Time 0.01979s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 45/ 50]	       MNIST	Time 0.01586s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 45/ 50]	FashionMNIST	Time 0.02512s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 46/ 50]	       MNIST	Time 0.01752s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 46/ 50]	FashionMNIST	Time 0.01713s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 47/ 50]	       MNIST	Time 0.02282s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 47/ 50]	FashionMNIST	Time 0.01728s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 48/ 50]	       MNIST	Time 0.02051s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 48/ 50]	FashionMNIST	Time 0.01513s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 49/ 50]	       MNIST	Time 0.01874s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 49/ 50]	FashionMNIST	Time 0.02564s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 50/ 50]	       MNIST	Time 0.01666s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 50/ 50]	FashionMNIST	Time 0.01730s	Training Accuracy: 100.00%	Test Accuracy: 78.12%

[FINAL]	       MNIST	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[FINAL]	FashionMNIST	Training Accuracy: 100.00%	Test Accuracy: 78.12%

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
