


<a id='MNIST-Classification-with-SimpleChains'></a>

# MNIST Classification with SimpleChains


SimpleChains.jl is an excellent framework for training small neural networks. In this tutorial we will demonstrate how to use the same API as Lux.jl to train a model using SimpleChains.jl. We will use the tutorial from [SimpleChains.jl](https://pumasai.github.io/SimpleChains.jl/dev/examples/mnist/) as a reference.


<a id='Package-Imports'></a>

## Package Imports


```julia
using Lux, MLUtils, Optimisers, Zygote, OneHotArrays, Random, Statistics
import MLDatasets: MNIST
import SimpleChains: static
```


<a id='Loading-MNIST'></a>

## Loading MNIST


```julia
function loadmnist(batchsize, train_split)
    # Load MNIST
    N = 2000
    dataset = MNIST(; split=:train)
    imgs = dataset.features[:, :, 1:N]
    labels_raw = dataset.targets[1:N]

    # Process images into (H,W,C,BS) batches
    x_data = Float32.(reshape(imgs, size(imgs, 1), size(imgs, 2), 1, size(imgs, 3)))
    y_data = onehotbatch(labels_raw, 0:9)
    (x_train, y_train), (x_test, y_test) = splitobs((x_data, y_data); at=train_split)

    return (
        # Use DataLoader to automatically minibatch and shuffle the data
        DataLoader(collect.((x_train, y_train)); batchsize, shuffle=true),
        # Don't shuffle the test data
        DataLoader(collect.((x_test, y_test)); batchsize, shuffle=false))
end
```


```
loadmnist (generic function with 1 method)
```


<a id='Define-the-Model'></a>

## Define the Model


```julia
lux_model = Chain(Conv((5, 5), 1 => 6, relu), MaxPool((2, 2)),
    Conv((5, 5), 6 => 16, relu), MaxPool((2, 2)), FlattenLayer(3),
    Chain(Dense(256 => 128, relu), Dense(128 => 84, relu), Dense(84 => 10)))
```


```
Chain(
    layer_1 = Conv((5, 5), 1 => 6, relu),  # 156 parameters
    layer_2 = MaxPool((2, 2)),
    layer_3 = Conv((5, 5), 6 => 16, relu),  # 2_416 parameters
    layer_4 = MaxPool((2, 2)),
    layer_5 = FlattenLayer(),
    layer_6 = Dense(256 => 128, relu),  # 32_896 parameters
    layer_7 = Dense(128 => 84, relu),   # 10_836 parameters
    layer_8 = Dense(84 => 10),          # 850 parameters
)         # Total: 47_154 parameters,
          #        plus 0 states.
```


We now need to convert the lux_model to SimpleChains.jl. We need to do this by defining the [`ToSimpleChainsAdaptor`](../../api/Lux/switching_frameworks#Lux.ToSimpleChainsAdaptor) and providing the input dimensions.


```julia
adaptor = ToSimpleChainsAdaptor((static(28), static(28), static(1)))
simple_chains_model = adaptor(lux_model)
```


```
SimpleChainsLayer()  # 47_154 parameters
```


<a id='Helper-Functions'></a>

## Helper Functions


```julia
logitcrossentropy(y_pred, y) = mean(-sum(y .* logsoftmax(y_pred); dims=1))

function loss(x, y, model, ps, st)
    y_pred, st = model(x, ps, st)
    return logitcrossentropy(y_pred, y), st
end

function accuracy(model, ps, st, dataloader)
    total_correct, total = 0, 0
    st = Lux.testmode(st)
    for (x, y) in dataloader
        target_class = onecold(y)
        predicted_class = onecold(Array(first(model(x, ps, st))))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end
```


```
accuracy (generic function with 1 method)
```


<a id='Define-the-Training-Loop'></a>

## Define the Training Loop


```julia
function train(model; rng=Xoshiro(0), kwargs...)
    ps, st = Lux.setup(rng, model)

    train_dataloader, test_dataloader = loadmnist(128, 0.9)
    opt = Adam(3.0f-4)
    st_opt = Optimisers.setup(opt, ps)

    ### Warmup the Model
    img = train_dataloader.data[1][:, :, :, 1:1]
    lab = train_dataloader.data[2][:, 1:1]
    loss(img, lab, model, ps, st)
    (l, _), back = pullback(p -> loss(img, lab, model, p, st), ps)
    back((one(l), nothing))

    ### Lets train the model
    nepochs = 9
    for epoch in 1:nepochs
        stime = time()
        for (x, y) in train_dataloader
            (l, st), back = pullback(loss, x, y, model, ps, st)
            ### We need to add `nothing`s equal to the number of returned values - 1
            gs = back((one(l), nothing))[4]
            st_opt, ps = Optimisers.update(st_opt, ps, gs)
        end
        ttime = time() - stime

        println("[$epoch/$nepochs] \t Time $(round(ttime; digits=2))s \t Training Accuracy: " *
                "$(round(accuracy(model, ps, st, train_dataloader) * 100; digits=2))% \t " *
                "Test Accuracy: $(round(accuracy(model, ps, st, test_dataloader) * 100; digits=2))%")
    end
end
```


```
train (generic function with 1 method)
```


<a id='Finally-Training-the-Model'></a>

## Finally Training the Model


First we will train the Lux model


```julia
train(lux_model)
```


```
[1/9] 	 Time 52.21s 	 Training Accuracy: 24.11% 	 Test Accuracy: 24.0%
[2/9] 	 Time 49.41s 	 Training Accuracy: 46.89% 	 Test Accuracy: 47.5%
[3/9] 	 Time 46.12s 	 Training Accuracy: 68.06% 	 Test Accuracy: 67.5%
[4/9] 	 Time 48.74s 	 Training Accuracy: 74.33% 	 Test Accuracy: 72.5%
[5/9] 	 Time 49.97s 	 Training Accuracy: 80.61% 	 Test Accuracy: 79.0%
[6/9] 	 Time 48.8s 	 Training Accuracy: 82.83% 	 Test Accuracy: 82.5%
[7/9] 	 Time 47.63s 	 Training Accuracy: 84.72% 	 Test Accuracy: 83.0%
[8/9] 	 Time 42.67s 	 Training Accuracy: 85.61% 	 Test Accuracy: 84.0%
[9/9] 	 Time 45.38s 	 Training Accuracy: 85.83% 	 Test Accuracy: 84.5%

```


Now we will train the SimpleChains model


```julia
train(simple_chains_model)
```


```
[1/9] 	 Time 16.51s 	 Training Accuracy: 43.11% 	 Test Accuracy: 39.5%
[2/9] 	 Time 15.76s 	 Training Accuracy: 55.33% 	 Test Accuracy: 49.5%
[3/9] 	 Time 15.75s 	 Training Accuracy: 66.94% 	 Test Accuracy: 64.0%
[4/9] 	 Time 15.76s 	 Training Accuracy: 76.67% 	 Test Accuracy: 74.5%
[5/9] 	 Time 15.76s 	 Training Accuracy: 78.78% 	 Test Accuracy: 75.5%
[6/9] 	 Time 15.79s 	 Training Accuracy: 84.83% 	 Test Accuracy: 84.0%
[7/9] 	 Time 15.76s 	 Training Accuracy: 87.11% 	 Test Accuracy: 85.0%
[8/9] 	 Time 15.75s 	 Training Accuracy: 88.22% 	 Test Accuracy: 87.0%
[9/9] 	 Time 15.76s 	 Training Accuracy: 88.44% 	 Test Accuracy: 87.0%

```


On my local machine we see a 3-4x speedup when using SimpleChains.jl. The conditions of the server this documentation is being built on is not ideal for CPU benchmarking hence, the speedup may not be as significant and even there might be regressions.


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
  JULIA_PROJECT = /var/lib/buildkite-agent/builds/gpuci-14/julialang/lux-dot-jl/docs/Project.toml
  JULIA_AMDGPU_LOGGING_ENABLED = true
  JULIA_DEBUG = Literate
  JULIA_CPU_THREADS = 2
  JULIA_NUM_THREADS = 48
  JULIA_LOAD_PATH = @:@v#.#:@stdlib
  JULIA_CUDA_HARD_MEMORY_LIMIT = 25%

```


---


*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

