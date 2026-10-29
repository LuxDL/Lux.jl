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
┌ Warning: Reactant unsafe_free! is not defined for OneHotArrays.OneHotMatrix{UInt32, Reactant.ConcretePJRTArray{UInt32, 1, 1}}.
└ @ ReactantExt ~/work/Lux.jl/Lux.jl/lib/MLDataDevices/ext/ReactantExt.jl:106
[  1/ 50]	       MNIST	Time 37.50574s	Training Accuracy: 34.57%	Test Accuracy: 37.50%
[  1/ 50]	FashionMNIST	Time 0.11836s	Training Accuracy: 32.62%	Test Accuracy: 43.75%
[  2/ 50]	       MNIST	Time 0.11264s	Training Accuracy: 36.04%	Test Accuracy: 31.25%
[  2/ 50]	FashionMNIST	Time 0.12775s	Training Accuracy: 46.58%	Test Accuracy: 50.00%
[  3/ 50]	       MNIST	Time 0.11024s	Training Accuracy: 43.36%	Test Accuracy: 28.12%
[  3/ 50]	FashionMNIST	Time 0.11044s	Training Accuracy: 57.13%	Test Accuracy: 56.25%
[  4/ 50]	       MNIST	Time 0.11195s	Training Accuracy: 51.07%	Test Accuracy: 40.62%
[  4/ 50]	FashionMNIST	Time 0.11200s	Training Accuracy: 63.96%	Test Accuracy: 62.50%
[  5/ 50]	       MNIST	Time 0.10969s	Training Accuracy: 56.93%	Test Accuracy: 31.25%
[  5/ 50]	FashionMNIST	Time 0.11141s	Training Accuracy: 70.02%	Test Accuracy: 53.12%
[  6/ 50]	       MNIST	Time 0.11026s	Training Accuracy: 64.06%	Test Accuracy: 37.50%
[  6/ 50]	FashionMNIST	Time 0.11806s	Training Accuracy: 74.71%	Test Accuracy: 56.25%
[  7/ 50]	       MNIST	Time 0.10915s	Training Accuracy: 69.63%	Test Accuracy: 46.88%
[  7/ 50]	FashionMNIST	Time 0.11132s	Training Accuracy: 77.15%	Test Accuracy: 62.50%
[  8/ 50]	       MNIST	Time 0.11133s	Training Accuracy: 76.66%	Test Accuracy: 46.88%
[  8/ 50]	FashionMNIST	Time 0.11220s	Training Accuracy: 81.15%	Test Accuracy: 65.62%
[  9/ 50]	       MNIST	Time 0.12861s	Training Accuracy: 80.66%	Test Accuracy: 43.75%
[  9/ 50]	FashionMNIST	Time 0.11198s	Training Accuracy: 83.20%	Test Accuracy: 65.62%
[ 10/ 50]	       MNIST	Time 0.11126s	Training Accuracy: 82.52%	Test Accuracy: 50.00%
[ 10/ 50]	FashionMNIST	Time 0.11561s	Training Accuracy: 89.06%	Test Accuracy: 59.38%
[ 11/ 50]	       MNIST	Time 0.11183s	Training Accuracy: 87.70%	Test Accuracy: 53.12%
[ 11/ 50]	FashionMNIST	Time 0.10971s	Training Accuracy: 89.45%	Test Accuracy: 65.62%
[ 12/ 50]	       MNIST	Time 0.11142s	Training Accuracy: 90.82%	Test Accuracy: 46.88%
[ 12/ 50]	FashionMNIST	Time 0.10974s	Training Accuracy: 91.80%	Test Accuracy: 68.75%
[ 13/ 50]	       MNIST	Time 0.11615s	Training Accuracy: 93.46%	Test Accuracy: 59.38%
[ 13/ 50]	FashionMNIST	Time 0.11101s	Training Accuracy: 91.60%	Test Accuracy: 68.75%
[ 14/ 50]	       MNIST	Time 0.11121s	Training Accuracy: 93.75%	Test Accuracy: 59.38%
[ 14/ 50]	FashionMNIST	Time 0.11157s	Training Accuracy: 94.34%	Test Accuracy: 68.75%
[ 15/ 50]	       MNIST	Time 0.11004s	Training Accuracy: 94.92%	Test Accuracy: 62.50%
[ 15/ 50]	FashionMNIST	Time 0.11527s	Training Accuracy: 91.89%	Test Accuracy: 65.62%
[ 16/ 50]	       MNIST	Time 0.11131s	Training Accuracy: 97.27%	Test Accuracy: 68.75%
[ 16/ 50]	FashionMNIST	Time 0.11067s	Training Accuracy: 95.80%	Test Accuracy: 68.75%
[ 17/ 50]	       MNIST	Time 0.11422s	Training Accuracy: 99.02%	Test Accuracy: 62.50%
[ 17/ 50]	FashionMNIST	Time 0.11020s	Training Accuracy: 96.68%	Test Accuracy: 68.75%
[ 18/ 50]	       MNIST	Time 0.11360s	Training Accuracy: 99.51%	Test Accuracy: 65.62%
[ 18/ 50]	FashionMNIST	Time 0.11274s	Training Accuracy: 96.88%	Test Accuracy: 68.75%
[ 19/ 50]	       MNIST	Time 0.11057s	Training Accuracy: 99.71%	Test Accuracy: 65.62%
[ 19/ 50]	FashionMNIST	Time 0.11561s	Training Accuracy: 98.54%	Test Accuracy: 71.88%
[ 20/ 50]	       MNIST	Time 0.11075s	Training Accuracy: 99.90%	Test Accuracy: 59.38%
[ 20/ 50]	FashionMNIST	Time 0.10916s	Training Accuracy: 98.54%	Test Accuracy: 68.75%
[ 21/ 50]	       MNIST	Time 0.11037s	Training Accuracy: 99.90%	Test Accuracy: 68.75%
[ 21/ 50]	FashionMNIST	Time 0.11076s	Training Accuracy: 98.54%	Test Accuracy: 75.00%
[ 22/ 50]	       MNIST	Time 0.11011s	Training Accuracy: 99.90%	Test Accuracy: 68.75%
[ 22/ 50]	FashionMNIST	Time 0.11106s	Training Accuracy: 99.51%	Test Accuracy: 71.88%
[ 23/ 50]	       MNIST	Time 0.10933s	Training Accuracy: 99.90%	Test Accuracy: 68.75%
[ 23/ 50]	FashionMNIST	Time 0.11692s	Training Accuracy: 99.71%	Test Accuracy: 68.75%
[ 24/ 50]	       MNIST	Time 0.10920s	Training Accuracy: 99.90%	Test Accuracy: 65.62%
[ 24/ 50]	FashionMNIST	Time 0.10977s	Training Accuracy: 99.80%	Test Accuracy: 68.75%
[ 25/ 50]	       MNIST	Time 0.11149s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 25/ 50]	FashionMNIST	Time 0.11082s	Training Accuracy: 99.80%	Test Accuracy: 71.88%
[ 26/ 50]	       MNIST	Time 0.11514s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 26/ 50]	FashionMNIST	Time 0.11212s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 27/ 50]	       MNIST	Time 0.11204s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 27/ 50]	FashionMNIST	Time 0.11024s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 28/ 50]	       MNIST	Time 0.11130s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 28/ 50]	FashionMNIST	Time 0.11113s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 29/ 50]	       MNIST	Time 0.11114s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 29/ 50]	FashionMNIST	Time 0.11143s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 30/ 50]	       MNIST	Time 0.11416s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 30/ 50]	FashionMNIST	Time 0.11036s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 31/ 50]	       MNIST	Time 0.10977s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 31/ 50]	FashionMNIST	Time 0.11030s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 32/ 50]	       MNIST	Time 0.11059s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 32/ 50]	FashionMNIST	Time 0.11802s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 33/ 50]	       MNIST	Time 0.11078s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 33/ 50]	FashionMNIST	Time 0.11190s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 34/ 50]	       MNIST	Time 0.11697s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 34/ 50]	FashionMNIST	Time 0.11141s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 35/ 50]	       MNIST	Time 0.11104s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 35/ 50]	FashionMNIST	Time 0.10954s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 36/ 50]	       MNIST	Time 0.11122s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 36/ 50]	FashionMNIST	Time 0.11596s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 37/ 50]	       MNIST	Time 0.11051s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 37/ 50]	FashionMNIST	Time 0.11035s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 38/ 50]	       MNIST	Time 0.11171s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 38/ 50]	FashionMNIST	Time 0.11122s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 39/ 50]	       MNIST	Time 0.11602s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 39/ 50]	FashionMNIST	Time 0.11024s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 40/ 50]	       MNIST	Time 0.11229s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 40/ 50]	FashionMNIST	Time 0.11538s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 41/ 50]	       MNIST	Time 0.11023s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 41/ 50]	FashionMNIST	Time 0.11043s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 42/ 50]	       MNIST	Time 0.10870s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 42/ 50]	FashionMNIST	Time 0.10922s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 43/ 50]	       MNIST	Time 0.11608s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 43/ 50]	FashionMNIST	Time 0.11980s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 44/ 50]	       MNIST	Time 0.11155s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 44/ 50]	FashionMNIST	Time 0.11187s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 45/ 50]	       MNIST	Time 0.10984s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 45/ 50]	FashionMNIST	Time 0.11011s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 46/ 50]	       MNIST	Time 0.10996s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 46/ 50]	FashionMNIST	Time 0.11076s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 47/ 50]	       MNIST	Time 0.11426s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 47/ 50]	FashionMNIST	Time 0.11159s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 48/ 50]	       MNIST	Time 0.11053s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 48/ 50]	FashionMNIST	Time 0.11024s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 49/ 50]	       MNIST	Time 0.10955s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 49/ 50]	FashionMNIST	Time 0.11671s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 50/ 50]	       MNIST	Time 0.11036s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 50/ 50]	FashionMNIST	Time 0.11084s	Training Accuracy: 100.00%	Test Accuracy: 68.75%

[FINAL]	       MNIST	Training Accuracy: 100.00%	Test Accuracy: 68.75%
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
