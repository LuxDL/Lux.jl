


<a id='Training-a-HyperNetwork-on-MNIST-and-FashionMNIST'></a>

# Training a HyperNetwork on MNIST and FashionMNIST


<a id='Package-Imports'></a>

## Package Imports


```julia
using Lux, ADTypes, ComponentArrays, LuxAMDGPU, LuxCUDA, MLDatasets, MLUtils, OneHotArrays,
      Optimisers, Printf, Random, Setfield, Statistics, Zygote

CUDA.allowscalar(false)
```


<a id='Loading-Datasets'></a>

## Loading Datasets


```julia
function load_dataset(::Type{dset}, n_train::Int, n_eval::Int, batchsize::Int) where {dset}
    imgs, labels = dset(:train)[1:n_train]
    x_train, y_train = reshape(imgs, 28, 28, 1, n_train), onehotbatch(labels, 0:9)

    imgs, labels = dset(:test)[1:n_eval]
    x_test, y_test = reshape(imgs, 28, 28, 1, n_eval), onehotbatch(labels, 0:9)

    return (DataLoader((x_train, y_train); batchsize=min(batchsize, n_train), shuffle=true),
        DataLoader((x_test, y_test); batchsize=min(batchsize, n_eval), shuffle=false))
end

function load_datasets(n_train=1024, n_eval=32, batchsize=256)
    return load_dataset.((MNIST, FashionMNIST), n_train, n_eval, batchsize)
end
```


```
load_datasets (generic function with 4 methods)
```


<a id='Implement-a-HyperNet-Layer'></a>

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


<a id='Create-and-Initialize-the-HyperNet'></a>

## Create and Initialize the HyperNet


```julia
function create_model()
    # Doesn't need to be a MLP can have any Lux Layer
    core_network = Chain(FlattenLayer(), Dense(784, 256, relu), Dense(256, 10))
    weight_generator = Chain(Embedding(2 => 32), Dense(32, 64, relu),
        Dense(64, Lux.parameterlength(core_network)))

    model = HyperNet(weight_generator, core_network)
    return model
end
```


```
create_model (generic function with 1 method)
```


<a id='Define-Utility-Functions'></a>

## Define Utility Functions


```julia
logitcrossentropy(y_pred, y) = mean(-sum(y .* logsoftmax(y_pred); dims=1))

function loss(model, ps, st, (data_idx, x, y))
    y_pred, st = model((data_idx, x), ps, st)
    return logitcrossentropy(y_pred, y), st, (;)
end

function accuracy(model, ps, st, dataloader, data_idx, gdev=gpu_device())
    total_correct, total = 0, 0
    st = Lux.testmode(st)
    cpu_dev = cpu_device()
    for (x, y) in dataloader
        x = x |> gdev
        y = y |> gdev
        target_class = onecold(cpu_dev(y))
        predicted_class = onecold(cpu_dev(model((data_idx, x), ps, st)[1]))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end
```


```
accuracy (generic function with 2 methods)
```


<a id='Training'></a>

## Training


```julia
function train()
    model = create_model()
    dataloaders = load_datasets()

    dev = gpu_device()

    rng = Xoshiro(0)

    train_state = Lux.Experimental.TrainState(
        rng, model, Adam(3.0f-4); transform_variables=dev)

    ### Lets train the model
    nepochs = 10
    for epoch in 1:nepochs, data_idx in 1:2
        train_dataloader, test_dataloader = dataloaders[data_idx]

        stime = time()
        for (x, y) in train_dataloader
            x = x |> dev
            y = y |> dev
            (gs, _, _, train_state) = Lux.Experimental.compute_gradients(
                AutoZygote(), loss, (data_idx, x, y), train_state)
            train_state = Lux.Experimental.apply_gradients(train_state, gs)
        end
        ttime = time() - stime

        train_acc = round(
            accuracy(model, train_state.parameters, train_state.states,
                train_dataloader, data_idx, dev) * 100;
            digits=2)
        test_acc = round(
            accuracy(model, train_state.parameters, train_state.states,
                test_dataloader, data_idx, dev) * 100;
            digits=2)

        data_name = data_idx == 1 ? "MNIST" : "FashionMNIST"

        @printf "[%3d/%3d] \t %12s \t Time %.5fs \t Training Accuracy: %.2f%% \t Test Accuracy: %.2f%%\n" epoch nepochs data_name ttime train_acc test_acc
    end

    println()

    for data_idx in 1:2
        train_dataloader, test_dataloader = dataloaders[data_idx]
        train_acc = round(
            accuracy(model, train_state.parameters, train_state.states,
                train_dataloader, data_idx, dev) * 100;
            digits=2)
        test_acc = round(
            accuracy(model, train_state.parameters, train_state.states,
                test_dataloader, data_idx, dev) * 100;
            digits=2)

        data_name = data_idx == 1 ? "MNIST" : "FashionMNIST"

        @printf "[FINAL] \t %12s \t Training Accuracy: %.2f%% \t Test Accuracy: %.2f%%\n" data_name train_acc test_acc
    end
end

train()
```


```
[  1/ 10] 	        MNIST 	 Time 65.15117s 	 Training Accuracy: 76.27% 	 Test Accuracy: 78.12%
[  1/ 10] 	 FashionMNIST 	 Time 0.15963s 	 Training Accuracy: 54.98% 	 Test Accuracy: 50.00%
[  2/ 10] 	        MNIST 	 Time 0.11049s 	 Training Accuracy: 75.49% 	 Test Accuracy: 71.88%
[  2/ 10] 	 FashionMNIST 	 Time 0.12701s 	 Training Accuracy: 56.45% 	 Test Accuracy: 65.62%
[  3/ 10] 	        MNIST 	 Time 0.03412s 	 Training Accuracy: 81.84% 	 Test Accuracy: 78.12%
[  3/ 10] 	 FashionMNIST 	 Time 0.03843s 	 Training Accuracy: 62.21% 	 Test Accuracy: 56.25%
[  4/ 10] 	        MNIST 	 Time 0.04174s 	 Training Accuracy: 82.62% 	 Test Accuracy: 81.25%
[  4/ 10] 	 FashionMNIST 	 Time 0.05071s 	 Training Accuracy: 66.41% 	 Test Accuracy: 56.25%
[  5/ 10] 	        MNIST 	 Time 0.03643s 	 Training Accuracy: 81.54% 	 Test Accuracy: 81.25%
[  5/ 10] 	 FashionMNIST 	 Time 0.05229s 	 Training Accuracy: 65.72% 	 Test Accuracy: 71.88%
[  6/ 10] 	        MNIST 	 Time 0.04656s 	 Training Accuracy: 90.53% 	 Test Accuracy: 90.62%
[  6/ 10] 	 FashionMNIST 	 Time 0.07645s 	 Training Accuracy: 69.14% 	 Test Accuracy: 62.50%
[  7/ 10] 	        MNIST 	 Time 0.03603s 	 Training Accuracy: 92.68% 	 Test Accuracy: 90.62%
[  7/ 10] 	 FashionMNIST 	 Time 0.02966s 	 Training Accuracy: 75.10% 	 Test Accuracy: 68.75%
[  8/ 10] 	        MNIST 	 Time 0.03106s 	 Training Accuracy: 93.85% 	 Test Accuracy: 90.62%
[  8/ 10] 	 FashionMNIST 	 Time 0.03912s 	 Training Accuracy: 74.02% 	 Test Accuracy: 71.88%
[  9/ 10] 	        MNIST 	 Time 0.03654s 	 Training Accuracy: 94.53% 	 Test Accuracy: 93.75%
[  9/ 10] 	 FashionMNIST 	 Time 0.03688s 	 Training Accuracy: 76.76% 	 Test Accuracy: 71.88%
[ 10/ 10] 	        MNIST 	 Time 0.04526s 	 Training Accuracy: 94.73% 	 Test Accuracy: 87.50%
[ 10/ 10] 	 FashionMNIST 	 Time 0.03781s 	 Training Accuracy: 80.08% 	 Test Accuracy: 65.62%

[FINAL] 	        MNIST 	 Training Accuracy: 91.70% 	 Test Accuracy: 78.12%
[FINAL] 	 FashionMNIST 	 Training Accuracy: 80.08% 	 Test Accuracy: 65.62%

```


<a id='Appendix'></a>

## Appendix


```julia
using InteractiveUtils
InteractiveUtils.versioninfo()
if @isdefined(LuxCUDA) && CUDA.functional(); println(); CUDA.versioninfo(); end
if @isdefined(LuxAMDGPU) && LuxAMDGPU.functional(); println(); AMDGPU.versioninfo(); end
```


```
Julia Version 1.10.2
Commit bd47eca2c8a (2024-03-01 10:14 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 48 × AMD EPYC 7402 24-Core Processor
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-15.0.7 (ORCJIT, znver2)
Threads: 48 default, 0 interactive, 24 GC (on 2 virtual cores)
Environment:
  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64
  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6
  JULIA_PROJECT = /var/lib/buildkite-agent/builds/gpuci-11/julialang/lux-dot-jl/docs/Project.toml
  JULIA_AMDGPU_LOGGING_ENABLED = true
  JULIA_DEBUG = Literate
  JULIA_CPU_THREADS = 2
  JULIA_NUM_THREADS = 48
  JULIA_LOAD_PATH = @:@v#.#:@stdlib
  JULIA_CUDA_HARD_MEMORY_LIMIT = 25%

CUDA runtime 12.3, artifact installation
CUDA driver 12.4
NVIDIA driver 550.54.14

CUDA libraries: 
- CUBLAS: 12.3.4
- CURAND: 10.3.4
- CUFFT: 11.0.12
- CUSOLVER: 11.5.4
- CUSPARSE: 12.2.0
- CUPTI: 21.0.0
- NVML: 12.0.0+550.54.14

Julia packages: 
- CUDA: 5.2.0
- CUDA_Driver_jll: 0.7.0+1
- CUDA_Runtime_jll: 0.11.1+0

Toolchain:
- Julia: 1.10.2
- LLVM: 15.0.7

Environment:
- JULIA_CUDA_HARD_MEMORY_LIMIT: 25%

1 device:
  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 2.620 GiB / 4.750 GiB available)
┌ Warning: LuxAMDGPU is loaded but the AMDGPU is not functional.
└ @ LuxAMDGPU ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/LuxAMDGPU/sGa0S/src/LuxAMDGPU.jl:19

```


---


*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
