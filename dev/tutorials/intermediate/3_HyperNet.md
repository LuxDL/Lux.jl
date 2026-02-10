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
[  1/ 50]	       MNIST	Time 44.97625s	Training Accuracy: 34.57%	Test Accuracy: 37.50%
[  1/ 50]	FashionMNIST	Time 0.10798s	Training Accuracy: 32.52%	Test Accuracy: 43.75%
[  2/ 50]	       MNIST	Time 0.11212s	Training Accuracy: 36.72%	Test Accuracy: 34.38%
[  2/ 50]	FashionMNIST	Time 0.13080s	Training Accuracy: 45.31%	Test Accuracy: 50.00%
[  3/ 50]	       MNIST	Time 0.10867s	Training Accuracy: 42.09%	Test Accuracy: 28.12%
[  3/ 50]	FashionMNIST	Time 0.11046s	Training Accuracy: 57.52%	Test Accuracy: 56.25%
[  4/ 50]	       MNIST	Time 0.12925s	Training Accuracy: 52.64%	Test Accuracy: 43.75%
[  4/ 50]	FashionMNIST	Time 0.10995s	Training Accuracy: 64.45%	Test Accuracy: 62.50%
[  5/ 50]	       MNIST	Time 0.10840s	Training Accuracy: 57.62%	Test Accuracy: 37.50%
[  5/ 50]	FashionMNIST	Time 0.10825s	Training Accuracy: 71.09%	Test Accuracy: 56.25%
[  6/ 50]	       MNIST	Time 0.10815s	Training Accuracy: 63.18%	Test Accuracy: 34.38%
[  6/ 50]	FashionMNIST	Time 0.11181s	Training Accuracy: 76.46%	Test Accuracy: 56.25%
[  7/ 50]	       MNIST	Time 0.10984s	Training Accuracy: 69.43%	Test Accuracy: 34.38%
[  7/ 50]	FashionMNIST	Time 0.11178s	Training Accuracy: 76.56%	Test Accuracy: 53.12%
[  8/ 50]	       MNIST	Time 0.11113s	Training Accuracy: 74.41%	Test Accuracy: 43.75%
[  8/ 50]	FashionMNIST	Time 0.10867s	Training Accuracy: 80.76%	Test Accuracy: 65.62%
[  9/ 50]	       MNIST	Time 0.11734s	Training Accuracy: 81.05%	Test Accuracy: 46.88%
[  9/ 50]	FashionMNIST	Time 0.10797s	Training Accuracy: 84.28%	Test Accuracy: 62.50%
[ 10/ 50]	       MNIST	Time 0.11054s	Training Accuracy: 83.50%	Test Accuracy: 50.00%
[ 10/ 50]	FashionMNIST	Time 0.11808s	Training Accuracy: 88.38%	Test Accuracy: 59.38%
[ 11/ 50]	       MNIST	Time 0.10875s	Training Accuracy: 88.28%	Test Accuracy: 53.12%
[ 11/ 50]	FashionMNIST	Time 0.10920s	Training Accuracy: 90.33%	Test Accuracy: 71.88%
[ 12/ 50]	       MNIST	Time 0.10751s	Training Accuracy: 90.33%	Test Accuracy: 53.12%
[ 12/ 50]	FashionMNIST	Time 0.11139s	Training Accuracy: 91.21%	Test Accuracy: 68.75%
[ 13/ 50]	       MNIST	Time 0.10940s	Training Accuracy: 94.14%	Test Accuracy: 59.38%
[ 13/ 50]	FashionMNIST	Time 0.10639s	Training Accuracy: 93.55%	Test Accuracy: 68.75%
[ 14/ 50]	       MNIST	Time 0.10975s	Training Accuracy: 95.41%	Test Accuracy: 56.25%
[ 14/ 50]	FashionMNIST	Time 0.11586s	Training Accuracy: 94.63%	Test Accuracy: 65.62%
[ 15/ 50]	       MNIST	Time 0.11185s	Training Accuracy: 95.31%	Test Accuracy: 59.38%
[ 15/ 50]	FashionMNIST	Time 0.12062s	Training Accuracy: 95.80%	Test Accuracy: 68.75%
[ 16/ 50]	       MNIST	Time 0.10447s	Training Accuracy: 98.54%	Test Accuracy: 53.12%
[ 16/ 50]	FashionMNIST	Time 0.10529s	Training Accuracy: 96.00%	Test Accuracy: 71.88%
[ 17/ 50]	       MNIST	Time 0.10438s	Training Accuracy: 99.32%	Test Accuracy: 59.38%
[ 17/ 50]	FashionMNIST	Time 0.10439s	Training Accuracy: 97.46%	Test Accuracy: 71.88%
[ 18/ 50]	       MNIST	Time 0.10251s	Training Accuracy: 99.61%	Test Accuracy: 59.38%
[ 18/ 50]	FashionMNIST	Time 0.10506s	Training Accuracy: 97.46%	Test Accuracy: 68.75%
[ 19/ 50]	       MNIST	Time 0.10619s	Training Accuracy: 99.90%	Test Accuracy: 53.12%
[ 19/ 50]	FashionMNIST	Time 0.10571s	Training Accuracy: 98.73%	Test Accuracy: 71.88%
[ 20/ 50]	       MNIST	Time 0.10402s	Training Accuracy: 99.90%	Test Accuracy: 50.00%
[ 20/ 50]	FashionMNIST	Time 0.11437s	Training Accuracy: 98.73%	Test Accuracy: 68.75%
[ 21/ 50]	       MNIST	Time 0.10730s	Training Accuracy: 99.90%	Test Accuracy: 53.12%
[ 21/ 50]	FashionMNIST	Time 0.10566s	Training Accuracy: 99.02%	Test Accuracy: 71.88%
[ 22/ 50]	       MNIST	Time 0.11882s	Training Accuracy: 99.90%	Test Accuracy: 56.25%
[ 22/ 50]	FashionMNIST	Time 0.10719s	Training Accuracy: 99.51%	Test Accuracy: 65.62%
[ 23/ 50]	       MNIST	Time 0.10671s	Training Accuracy: 99.90%	Test Accuracy: 53.12%
[ 23/ 50]	FashionMNIST	Time 0.10430s	Training Accuracy: 99.71%	Test Accuracy: 68.75%
[ 24/ 50]	       MNIST	Time 0.10453s	Training Accuracy: 99.90%	Test Accuracy: 56.25%
[ 24/ 50]	FashionMNIST	Time 0.10569s	Training Accuracy: 99.90%	Test Accuracy: 68.75%
[ 25/ 50]	       MNIST	Time 0.10590s	Training Accuracy: 99.90%	Test Accuracy: 53.12%
[ 25/ 50]	FashionMNIST	Time 0.10588s	Training Accuracy: 99.90%	Test Accuracy: 68.75%
[ 26/ 50]	       MNIST	Time 0.10628s	Training Accuracy: 100.00%	Test Accuracy: 50.00%
[ 26/ 50]	FashionMNIST	Time 0.10396s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 27/ 50]	       MNIST	Time 0.11474s	Training Accuracy: 100.00%	Test Accuracy: 50.00%
[ 27/ 50]	FashionMNIST	Time 0.10845s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 28/ 50]	       MNIST	Time 0.10815s	Training Accuracy: 100.00%	Test Accuracy: 53.12%
[ 28/ 50]	FashionMNIST	Time 0.11219s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 29/ 50]	       MNIST	Time 0.11023s	Training Accuracy: 100.00%	Test Accuracy: 50.00%
[ 29/ 50]	FashionMNIST	Time 0.10604s	Training Accuracy: 100.00%	Test Accuracy: 65.62%
[ 30/ 50]	       MNIST	Time 0.10760s	Training Accuracy: 100.00%	Test Accuracy: 53.12%
[ 30/ 50]	FashionMNIST	Time 0.10661s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 31/ 50]	       MNIST	Time 0.10480s	Training Accuracy: 100.00%	Test Accuracy: 53.12%
[ 31/ 50]	FashionMNIST	Time 0.11500s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 32/ 50]	       MNIST	Time 0.11997s	Training Accuracy: 100.00%	Test Accuracy: 53.12%
[ 32/ 50]	FashionMNIST	Time 0.10705s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 33/ 50]	       MNIST	Time 0.10516s	Training Accuracy: 100.00%	Test Accuracy: 53.12%
[ 33/ 50]	FashionMNIST	Time 0.11574s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 34/ 50]	       MNIST	Time 0.10734s	Training Accuracy: 100.00%	Test Accuracy: 53.12%
[ 34/ 50]	FashionMNIST	Time 0.10587s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 35/ 50]	       MNIST	Time 0.10780s	Training Accuracy: 100.00%	Test Accuracy: 53.12%
[ 35/ 50]	FashionMNIST	Time 0.10754s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 36/ 50]	       MNIST	Time 0.10757s	Training Accuracy: 100.00%	Test Accuracy: 53.12%
[ 36/ 50]	FashionMNIST	Time 0.10507s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 37/ 50]	       MNIST	Time 0.10895s	Training Accuracy: 100.00%	Test Accuracy: 53.12%
[ 37/ 50]	FashionMNIST	Time 0.10495s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 38/ 50]	       MNIST	Time 0.10884s	Training Accuracy: 100.00%	Test Accuracy: 56.25%
[ 38/ 50]	FashionMNIST	Time 0.11944s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 39/ 50]	       MNIST	Time 0.10709s	Training Accuracy: 100.00%	Test Accuracy: 56.25%
[ 39/ 50]	FashionMNIST	Time 0.10873s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 40/ 50]	       MNIST	Time 0.11872s	Training Accuracy: 100.00%	Test Accuracy: 53.12%
[ 40/ 50]	FashionMNIST	Time 0.10627s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 41/ 50]	       MNIST	Time 0.10567s	Training Accuracy: 100.00%	Test Accuracy: 56.25%
[ 41/ 50]	FashionMNIST	Time 0.10502s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 42/ 50]	       MNIST	Time 0.11189s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 42/ 50]	FashionMNIST	Time 0.10648s	Training Accuracy: 100.00%	Test Accuracy: 71.88%
[ 43/ 50]	       MNIST	Time 0.10399s	Training Accuracy: 100.00%	Test Accuracy: 56.25%
[ 43/ 50]	FashionMNIST	Time 0.10416s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 44/ 50]	       MNIST	Time 0.10518s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 44/ 50]	FashionMNIST	Time 0.10665s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 45/ 50]	       MNIST	Time 0.11418s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 45/ 50]	FashionMNIST	Time 0.10518s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 46/ 50]	       MNIST	Time 0.10581s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 46/ 50]	FashionMNIST	Time 0.11420s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 47/ 50]	       MNIST	Time 0.10915s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 47/ 50]	FashionMNIST	Time 0.10690s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 48/ 50]	       MNIST	Time 0.10466s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 48/ 50]	FashionMNIST	Time 0.11144s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 49/ 50]	       MNIST	Time 0.10410s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 49/ 50]	FashionMNIST	Time 0.10250s	Training Accuracy: 100.00%	Test Accuracy: 68.75%
[ 50/ 50]	       MNIST	Time 0.11605s	Training Accuracy: 100.00%	Test Accuracy: 59.38%
[ 50/ 50]	FashionMNIST	Time 0.10277s	Training Accuracy: 100.00%	Test Accuracy: 68.75%

[FINAL]	       MNIST	Training Accuracy: 100.00%	Test Accuracy: 59.38%
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
Julia Version 1.12.4
Commit 01a2eadb047 (2026-01-06 16:56 UTC)
Build Info:
  Official https://julialang.org release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 4 × Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, icelake-server)
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
