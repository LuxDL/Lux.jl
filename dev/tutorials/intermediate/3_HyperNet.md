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
[  1/ 50]	       MNIST	Time 23.21210s	Training Accuracy: 35.35%	Test Accuracy: 37.50%
[  1/ 50]	FashionMNIST	Time 0.01166s	Training Accuracy: 32.81%	Test Accuracy: 43.75%
[  2/ 50]	       MNIST	Time 0.04230s	Training Accuracy: 41.70%	Test Accuracy: 40.62%
[  2/ 50]	FashionMNIST	Time 0.01041s	Training Accuracy: 58.20%	Test Accuracy: 65.62%
[  3/ 50]	       MNIST	Time 0.01090s	Training Accuracy: 56.05%	Test Accuracy: 43.75%
[  3/ 50]	FashionMNIST	Time 0.02327s	Training Accuracy: 68.36%	Test Accuracy: 68.75%
[  4/ 50]	       MNIST	Time 0.01307s	Training Accuracy: 66.70%	Test Accuracy: 56.25%
[  4/ 50]	FashionMNIST	Time 0.02175s	Training Accuracy: 74.22%	Test Accuracy: 65.62%
[  5/ 50]	       MNIST	Time 0.01129s	Training Accuracy: 75.78%	Test Accuracy: 56.25%
[  5/ 50]	FashionMNIST	Time 0.01129s	Training Accuracy: 78.32%	Test Accuracy: 65.62%
[  6/ 50]	       MNIST	Time 0.01077s	Training Accuracy: 80.18%	Test Accuracy: 56.25%
[  6/ 50]	FashionMNIST	Time 0.01390s	Training Accuracy: 82.71%	Test Accuracy: 68.75%
[  7/ 50]	       MNIST	Time 0.01079s	Training Accuracy: 84.96%	Test Accuracy: 56.25%
[  7/ 50]	FashionMNIST	Time 0.00981s	Training Accuracy: 84.47%	Test Accuracy: 68.75%
[  8/ 50]	       MNIST	Time 0.01657s	Training Accuracy: 89.55%	Test Accuracy: 59.38%
[  8/ 50]	FashionMNIST	Time 0.01143s	Training Accuracy: 87.21%	Test Accuracy: 75.00%
[  9/ 50]	       MNIST	Time 0.02259s	Training Accuracy: 92.29%	Test Accuracy: 71.88%
[  9/ 50]	FashionMNIST	Time 0.01082s	Training Accuracy: 91.60%	Test Accuracy: 68.75%
[ 10/ 50]	       MNIST	Time 0.02259s	Training Accuracy: 95.70%	Test Accuracy: 71.88%
[ 10/ 50]	FashionMNIST	Time 0.01081s	Training Accuracy: 92.19%	Test Accuracy: 68.75%
[ 11/ 50]	       MNIST	Time 0.00867s	Training Accuracy: 96.68%	Test Accuracy: 75.00%
[ 11/ 50]	FashionMNIST	Time 0.00978s	Training Accuracy: 93.16%	Test Accuracy: 75.00%
[ 12/ 50]	       MNIST	Time 0.01028s	Training Accuracy: 98.73%	Test Accuracy: 75.00%
[ 12/ 50]	FashionMNIST	Time 0.01079s	Training Accuracy: 95.51%	Test Accuracy: 71.88%
[ 13/ 50]	       MNIST	Time 0.01096s	Training Accuracy: 99.71%	Test Accuracy: 75.00%
[ 13/ 50]	FashionMNIST	Time 0.01066s	Training Accuracy: 96.39%	Test Accuracy: 78.12%
[ 14/ 50]	       MNIST	Time 0.01071s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 14/ 50]	FashionMNIST	Time 0.02130s	Training Accuracy: 98.63%	Test Accuracy: 78.12%
[ 15/ 50]	       MNIST	Time 0.01031s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 15/ 50]	FashionMNIST	Time 0.02089s	Training Accuracy: 99.12%	Test Accuracy: 78.12%
[ 16/ 50]	       MNIST	Time 0.01005s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 16/ 50]	FashionMNIST	Time 0.02314s	Training Accuracy: 99.61%	Test Accuracy: 81.25%
[ 17/ 50]	       MNIST	Time 0.01067s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 17/ 50]	FashionMNIST	Time 0.01176s	Training Accuracy: 97.66%	Test Accuracy: 75.00%
[ 18/ 50]	       MNIST	Time 0.01102s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 18/ 50]	FashionMNIST	Time 0.00968s	Training Accuracy: 99.71%	Test Accuracy: 75.00%
[ 19/ 50]	       MNIST	Time 0.01231s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 19/ 50]	FashionMNIST	Time 0.01082s	Training Accuracy: 99.71%	Test Accuracy: 71.88%
[ 20/ 50]	       MNIST	Time 0.01069s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 20/ 50]	FashionMNIST	Time 0.01058s	Training Accuracy: 99.90%	Test Accuracy: 78.12%
[ 21/ 50]	       MNIST	Time 0.02163s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 21/ 50]	FashionMNIST	Time 0.01057s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 22/ 50]	       MNIST	Time 0.01934s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 22/ 50]	FashionMNIST	Time 0.00949s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 23/ 50]	       MNIST	Time 0.02169s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 23/ 50]	FashionMNIST	Time 0.01608s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 24/ 50]	       MNIST	Time 0.00949s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 24/ 50]	FashionMNIST	Time 0.01063s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 25/ 50]	       MNIST	Time 0.01192s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 25/ 50]	FashionMNIST	Time 0.01438s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 26/ 50]	       MNIST	Time 0.01064s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 26/ 50]	FashionMNIST	Time 0.01148s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 27/ 50]	       MNIST	Time 0.01095s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 27/ 50]	FashionMNIST	Time 0.01930s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 28/ 50]	       MNIST	Time 0.01156s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 28/ 50]	FashionMNIST	Time 0.02321s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 29/ 50]	       MNIST	Time 0.01068s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 29/ 50]	FashionMNIST	Time 0.00968s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 30/ 50]	       MNIST	Time 0.01098s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 30/ 50]	FashionMNIST	Time 0.01027s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 31/ 50]	       MNIST	Time 0.00944s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 31/ 50]	FashionMNIST	Time 0.00934s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 32/ 50]	       MNIST	Time 0.01055s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 32/ 50]	FashionMNIST	Time 0.00956s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 33/ 50]	       MNIST	Time 0.02378s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 33/ 50]	FashionMNIST	Time 0.01104s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 34/ 50]	       MNIST	Time 0.02462s	Training Accuracy: 100.00%	Test Accuracy: 81.25%
[ 34/ 50]	FashionMNIST	Time 0.00984s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 35/ 50]	       MNIST	Time 0.02024s	Training Accuracy: 100.00%	Test Accuracy: 81.25%
[ 35/ 50]	FashionMNIST	Time 0.01002s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 36/ 50]	       MNIST	Time 0.01026s	Training Accuracy: 100.00%	Test Accuracy: 81.25%
[ 36/ 50]	FashionMNIST	Time 0.01225s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 37/ 50]	       MNIST	Time 0.01066s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 37/ 50]	FashionMNIST	Time 0.00940s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 38/ 50]	       MNIST	Time 0.01076s	Training Accuracy: 100.00%	Test Accuracy: 81.25%
[ 38/ 50]	FashionMNIST	Time 0.01370s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 39/ 50]	       MNIST	Time 0.01075s	Training Accuracy: 100.00%	Test Accuracy: 81.25%
[ 39/ 50]	FashionMNIST	Time 0.02069s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 40/ 50]	       MNIST	Time 0.01033s	Training Accuracy: 100.00%	Test Accuracy: 81.25%
[ 40/ 50]	FashionMNIST	Time 0.01998s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 41/ 50]	       MNIST	Time 0.01024s	Training Accuracy: 100.00%	Test Accuracy: 81.25%
[ 41/ 50]	FashionMNIST	Time 0.02362s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 42/ 50]	       MNIST	Time 0.01211s	Training Accuracy: 100.00%	Test Accuracy: 81.25%
[ 42/ 50]	FashionMNIST	Time 0.00865s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 43/ 50]	       MNIST	Time 0.01069s	Training Accuracy: 100.00%	Test Accuracy: 81.25%
[ 43/ 50]	FashionMNIST	Time 0.01482s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 44/ 50]	       MNIST	Time 0.01084s	Training Accuracy: 100.00%	Test Accuracy: 81.25%
[ 44/ 50]	FashionMNIST	Time 0.01020s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 45/ 50]	       MNIST	Time 0.01122s	Training Accuracy: 100.00%	Test Accuracy: 81.25%
[ 45/ 50]	FashionMNIST	Time 0.01082s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 46/ 50]	       MNIST	Time 0.02191s	Training Accuracy: 100.00%	Test Accuracy: 81.25%
[ 46/ 50]	FashionMNIST	Time 0.00942s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 47/ 50]	       MNIST	Time 0.01883s	Training Accuracy: 100.00%	Test Accuracy: 81.25%
[ 47/ 50]	FashionMNIST	Time 0.00939s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 48/ 50]	       MNIST	Time 0.01025s	Training Accuracy: 100.00%	Test Accuracy: 81.25%
[ 48/ 50]	FashionMNIST	Time 0.01223s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 49/ 50]	       MNIST	Time 0.01054s	Training Accuracy: 100.00%	Test Accuracy: 81.25%
[ 49/ 50]	FashionMNIST	Time 0.01080s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 50/ 50]	       MNIST	Time 0.01052s	Training Accuracy: 100.00%	Test Accuracy: 81.25%
[ 50/ 50]	FashionMNIST	Time 0.01055s	Training Accuracy: 100.00%	Test Accuracy: 75.00%

[FINAL]	       MNIST	Training Accuracy: 100.00%	Test Accuracy: 81.25%
[FINAL]	FashionMNIST	Training Accuracy: 100.00%	Test Accuracy: 75.00%

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
Julia Version 1.13.4
Commit c1ac32ab733 (2026-11-13 17:30 UTC)
Build Info:
  Official https://julialang.org release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 4 × AMD EPYC 9V45 96-Core Processor
  WORD_SIZE: 64
  LLVM: libLLVM-20.1.8 (ORCJIT, znver5)
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
