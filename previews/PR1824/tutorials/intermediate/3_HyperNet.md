---
url: /previews/PR1824/tutorials/intermediate/3_HyperNet.md
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
[  1/ 50]	       MNIST	Time 26.84049s	Training Accuracy: 34.28%	Test Accuracy: 37.50%
[  1/ 50]	FashionMNIST	Time 0.01400s	Training Accuracy: 36.91%	Test Accuracy: 43.75%
[  2/ 50]	       MNIST	Time 0.01790s	Training Accuracy: 46.78%	Test Accuracy: 40.62%
[  2/ 50]	FashionMNIST	Time 0.04029s	Training Accuracy: 58.59%	Test Accuracy: 56.25%
[  3/ 50]	       MNIST	Time 0.01349s	Training Accuracy: 59.18%	Test Accuracy: 43.75%
[  3/ 50]	FashionMNIST	Time 0.01261s	Training Accuracy: 69.24%	Test Accuracy: 65.62%
[  4/ 50]	       MNIST	Time 0.03137s	Training Accuracy: 69.63%	Test Accuracy: 56.25%
[  4/ 50]	FashionMNIST	Time 0.01052s	Training Accuracy: 73.83%	Test Accuracy: 68.75%
[  5/ 50]	       MNIST	Time 0.01117s	Training Accuracy: 77.73%	Test Accuracy: 59.38%
[  5/ 50]	FashionMNIST	Time 0.01015s	Training Accuracy: 77.15%	Test Accuracy: 65.62%
[  6/ 50]	       MNIST	Time 0.01061s	Training Accuracy: 81.45%	Test Accuracy: 56.25%
[  6/ 50]	FashionMNIST	Time 0.01085s	Training Accuracy: 82.03%	Test Accuracy: 68.75%
[  7/ 50]	       MNIST	Time 0.01133s	Training Accuracy: 87.79%	Test Accuracy: 62.50%
[  7/ 50]	FashionMNIST	Time 0.01300s	Training Accuracy: 83.59%	Test Accuracy: 65.62%
[  8/ 50]	       MNIST	Time 0.02197s	Training Accuracy: 91.50%	Test Accuracy: 59.38%
[  8/ 50]	FashionMNIST	Time 0.01283s	Training Accuracy: 88.67%	Test Accuracy: 68.75%
[  9/ 50]	       MNIST	Time 0.01613s	Training Accuracy: 91.21%	Test Accuracy: 62.50%
[  9/ 50]	FashionMNIST	Time 0.02706s	Training Accuracy: 91.21%	Test Accuracy: 71.88%
[ 10/ 50]	       MNIST	Time 0.01113s	Training Accuracy: 95.80%	Test Accuracy: 62.50%
[ 10/ 50]	FashionMNIST	Time 0.00970s	Training Accuracy: 94.34%	Test Accuracy: 75.00%
[ 11/ 50]	       MNIST	Time 0.01048s	Training Accuracy: 97.75%	Test Accuracy: 68.75%
[ 11/ 50]	FashionMNIST	Time 0.01012s	Training Accuracy: 95.31%	Test Accuracy: 75.00%
[ 12/ 50]	       MNIST	Time 0.00993s	Training Accuracy: 99.02%	Test Accuracy: 65.62%
[ 12/ 50]	FashionMNIST	Time 0.01236s	Training Accuracy: 96.19%	Test Accuracy: 78.12%
[ 13/ 50]	       MNIST	Time 0.00970s	Training Accuracy: 99.12%	Test Accuracy: 59.38%
[ 13/ 50]	FashionMNIST	Time 0.02218s	Training Accuracy: 96.88%	Test Accuracy: 71.88%
[ 14/ 50]	       MNIST	Time 0.00970s	Training Accuracy: 99.90%	Test Accuracy: 59.38%
[ 14/ 50]	FashionMNIST	Time 0.01240s	Training Accuracy: 98.05%	Test Accuracy: 81.25%
[ 15/ 50]	       MNIST	Time 0.01984s	Training Accuracy: 99.90%	Test Accuracy: 65.62%
[ 15/ 50]	FashionMNIST	Time 0.01307s	Training Accuracy: 99.12%	Test Accuracy: 75.00%
[ 16/ 50]	       MNIST	Time 0.01094s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 16/ 50]	FashionMNIST	Time 0.01025s	Training Accuracy: 98.93%	Test Accuracy: 75.00%
[ 17/ 50]	       MNIST	Time 0.01268s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 17/ 50]	FashionMNIST	Time 0.01078s	Training Accuracy: 98.73%	Test Accuracy: 78.12%
[ 18/ 50]	       MNIST	Time 0.01190s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 18/ 50]	FashionMNIST	Time 0.01156s	Training Accuracy: 98.54%	Test Accuracy: 75.00%
[ 19/ 50]	       MNIST	Time 0.02421s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 19/ 50]	FashionMNIST	Time 0.01062s	Training Accuracy: 99.71%	Test Accuracy: 75.00%
[ 20/ 50]	       MNIST	Time 0.01144s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 20/ 50]	FashionMNIST	Time 0.01210s	Training Accuracy: 99.02%	Test Accuracy: 75.00%
[ 21/ 50]	       MNIST	Time 0.00944s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 21/ 50]	FashionMNIST	Time 0.01266s	Training Accuracy: 99.90%	Test Accuracy: 75.00%
[ 22/ 50]	       MNIST	Time 0.01023s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 22/ 50]	FashionMNIST	Time 0.01078s	Training Accuracy: 99.90%	Test Accuracy: 75.00%
[ 23/ 50]	       MNIST	Time 0.02209s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 23/ 50]	FashionMNIST	Time 0.01235s	Training Accuracy: 99.90%	Test Accuracy: 75.00%
[ 24/ 50]	       MNIST	Time 0.01212s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 24/ 50]	FashionMNIST	Time 0.02818s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 25/ 50]	       MNIST	Time 0.00894s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 25/ 50]	FashionMNIST	Time 0.01264s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 26/ 50]	       MNIST	Time 0.01393s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 26/ 50]	FashionMNIST	Time 0.01021s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 27/ 50]	       MNIST	Time 0.01182s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 27/ 50]	FashionMNIST	Time 0.01098s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 28/ 50]	       MNIST	Time 0.01071s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 28/ 50]	FashionMNIST	Time 0.02240s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 29/ 50]	       MNIST	Time 0.00970s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 29/ 50]	FashionMNIST	Time 0.01019s	Training Accuracy: 100.00%	Test Accuracy: 78.12%
[ 30/ 50]	       MNIST	Time 0.02405s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 30/ 50]	FashionMNIST	Time 0.01087s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 31/ 50]	       MNIST	Time 0.01038s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 31/ 50]	FashionMNIST	Time 0.01073s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 32/ 50]	       MNIST	Time 0.00942s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 32/ 50]	FashionMNIST	Time 0.01100s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 33/ 50]	       MNIST	Time 0.01027s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 33/ 50]	FashionMNIST	Time 0.01050s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 34/ 50]	       MNIST	Time 0.02302s	Training Accuracy: 100.00%	Test Accuracy: 62.50%
[ 34/ 50]	FashionMNIST	Time 0.01360s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 35/ 50]	       MNIST	Time 0.00961s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 35/ 50]	FashionMNIST	Time 0.01050s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 36/ 50]	       MNIST	Time 0.00885s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 36/ 50]	FashionMNIST	Time 0.01056s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 37/ 50]	       MNIST	Time 0.01115s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 37/ 50]	FashionMNIST	Time 0.01090s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 38/ 50]	       MNIST	Time 0.02630s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 38/ 50]	FashionMNIST	Time 0.01171s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 39/ 50]	       MNIST	Time 0.01251s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 39/ 50]	FashionMNIST	Time 0.02417s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 40/ 50]	       MNIST	Time 0.01044s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 40/ 50]	FashionMNIST	Time 0.00878s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 41/ 50]	       MNIST	Time 0.01171s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 41/ 50]	FashionMNIST	Time 0.00888s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 42/ 50]	       MNIST	Time 0.01115s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 42/ 50]	FashionMNIST	Time 0.01163s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 43/ 50]	       MNIST	Time 0.01701s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 43/ 50]	FashionMNIST	Time 0.02433s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 44/ 50]	       MNIST	Time 0.00986s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 44/ 50]	FashionMNIST	Time 0.01072s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 45/ 50]	       MNIST	Time 0.02434s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 45/ 50]	FashionMNIST	Time 0.00993s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 46/ 50]	       MNIST	Time 0.01160s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 46/ 50]	FashionMNIST	Time 0.01341s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 47/ 50]	       MNIST	Time 0.01131s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 47/ 50]	FashionMNIST	Time 0.01021s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 48/ 50]	       MNIST	Time 0.01232s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 48/ 50]	FashionMNIST	Time 0.01079s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 49/ 50]	       MNIST	Time 0.02361s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 49/ 50]	FashionMNIST	Time 0.01114s	Training Accuracy: 100.00%	Test Accuracy: 75.00%
[ 50/ 50]	       MNIST	Time 0.01119s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 50/ 50]	FashionMNIST	Time 0.01890s	Training Accuracy: 100.00%	Test Accuracy: 75.00%

[FINAL]	       MNIST	Training Accuracy: 100.00%	Test Accuracy: 65.62%
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
Julia Version 1.12.7
Commit 6d172b025e4 (2026-08-15 08:05 UTC)
Build Info:
  Official https://julialang.org release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 4 × AMD EPYC 9V45 96-Core Processor
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, znver5)
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
