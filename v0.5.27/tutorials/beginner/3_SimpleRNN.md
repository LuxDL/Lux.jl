


<a id='Training-a-Simple-LSTM'></a>

# Training a Simple LSTM


In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:


1. Create custom Lux models.
2. Become familiar with the Lux recurrent neural network API.
3. Training using Optimisers.jl and Zygote.jl.


<a id='Package-Imports'></a>

## Package Imports


```julia
using Lux, LuxAMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Random, Statistics
```


<a id='Dataset'></a>

## Dataset


We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a `MLUtils.DataLoader`. Our dataloader will give us sequences of size 2 × seq*len × batch*size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.


```julia
function get_dataloaders(; dataset_size=1000, sequence_length=50)
    # Create the spirals
    data = [MLUtils.Datasets.make_spiral(sequence_length) for _ in 1:dataset_size]
    # Get the labels
    labels = vcat(repeat([0.0f0], dataset_size ÷ 2), repeat([1.0f0], dataset_size ÷ 2))
    clockwise_spirals = [reshape(d[1][:, 1:sequence_length], :, sequence_length, 1)
                         for d in data[1:(dataset_size ÷ 2)]]
    anticlockwise_spirals = [reshape(
                                 d[1][:, (sequence_length + 1):end], :, sequence_length, 1)
                             for d in data[((dataset_size ÷ 2) + 1):end]]
    x_data = Float32.(cat(clockwise_spirals..., anticlockwise_spirals...; dims=3))
    # Split the dataset
    (x_train, y_train), (x_val, y_val) = splitobs((x_data, labels); at=0.8, shuffle=true)
    # Create DataLoaders
    return (
        # Use DataLoader to automatically minibatch and shuffle the data
        DataLoader(collect.((x_train, y_train)); batchsize=128, shuffle=true),
        # Don't shuffle the validation data
        DataLoader(collect.((x_val, y_val)); batchsize=128, shuffle=false))
end
```


```
get_dataloaders (generic function with 1 method)
```


<a id='Creating-a-Classifier'></a>

## Creating a Classifier


We will be extending the `Lux.AbstractExplicitContainerLayer` type for our custom model since it will contain a lstm block and a classifier head.


We pass the fieldnames `lstm_cell` and `classifier` to the type to ensure that the parameters and states are automatically populated and we don't have to define `Lux.initialparameters` and `Lux.initialstates`.


To understand more about container layers, please look at [Container Layer](../../manual/interface#Container-Layer).


```julia
struct SpiralClassifier{L, C} <:
       Lux.AbstractExplicitContainerLayer{(:lstm_cell, :classifier)}
    lstm_cell::L
    classifier::C
end
```


We won't define the model from scratch but rather use the [`Lux.LSTMCell`](../../api/Lux/layers#Lux.LSTMCell) and [`Lux.Dense`](../../api/Lux/layers#Lux.Dense).


```julia
function SpiralClassifier(in_dims, hidden_dims, out_dims)
    return SpiralClassifier(
        LSTMCell(in_dims => hidden_dims), Dense(hidden_dims => out_dims, sigmoid))
end
```


```
Main.var"##225".SpiralClassifier
```


We can use default Lux blocks – `Recurrence(LSTMCell(in_dims => hidden_dims)` – instead of defining the following. But let's still do it for the sake of it.


Now we need to define the behavior of the Classifier when it is invoked.


```julia
function (s::SpiralClassifier)(
        x::AbstractArray{T, 3}, ps::NamedTuple, st::NamedTuple) where {T}
    # First we will have to run the sequence through the LSTM Cell
    # The first call to LSTM Cell will create the initial hidden state
    # See that the parameters and states are automatically populated into a field called
    # `lstm_cell` We use `eachslice` to get the elements in the sequence without copying,
    # and `Iterators.peel` to split out the first element for LSTM initialization.
    x_init, x_rest = Iterators.peel(Lux._eachslice(x, Val(2)))
    (y, carry), st_lstm = s.lstm_cell(x_init, ps.lstm_cell, st.lstm_cell)
    # Now that we have the hidden state and memory in `carry` we will pass the input and
    # `carry` jointly
    for x in x_rest
        (y, carry), st_lstm = s.lstm_cell((x, carry), ps.lstm_cell, st_lstm)
    end
    # After running through the sequence we will pass the output through the classifier
    y, st_classifier = s.classifier(y, ps.classifier, st.classifier)
    # Finally remember to create the updated state
    st = merge(st, (classifier=st_classifier, lstm_cell=st_lstm))
    return vec(y), st
end
```


<a id='Defining-Accuracy,-Loss-and-Optimiser'></a>

## Defining Accuracy, Loss and Optimiser


Now let's define the binarycrossentropy loss. Typically it is recommended to use `logitbinarycrossentropy` since it is more numerically stable, but for the sake of simplicity we will use `binarycrossentropy`.


```julia
function xlogy(x, y)
    result = x * log(y)
    return ifelse(iszero(x), zero(result), result)
end

function binarycrossentropy(y_pred, y_true)
    y_pred = y_pred .+ eps(eltype(y_pred))
    return mean(@. -xlogy(y_true, y_pred) - xlogy(1 - y_true, 1 - y_pred))
end

function compute_loss(x, y, model, ps, st)
    y_pred, st = model(x, ps, st)
    return binarycrossentropy(y_pred, y), y_pred, st
end

matches(y_pred, y_true) = sum((y_pred .> 0.5f0) .== y_true)
accuracy(y_pred, y_true) = matches(y_pred, y_true) / length(y_pred)
```


```
accuracy (generic function with 1 method)
```


Finally lets create an optimiser given the model parameters.


```julia
function create_optimiser(ps)
    opt = Optimisers.Adam(0.01f0)
    return Optimisers.setup(opt, ps)
end
```


```
create_optimiser (generic function with 1 method)
```


<a id='Training-the-Model'></a>

## Training the Model


```julia
function main()
    # Get the dataloaders
    (train_loader, val_loader) = get_dataloaders()

    # Create the model
    model = SpiralClassifier(2, 8, 1)
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    ps, st = Lux.setup(rng, model)

    dev = gpu_device()
    ps = ps |> dev
    st = st |> dev

    # Create the optimiser
    opt_state = create_optimiser(ps)

    for epoch in 1:25
        # Train the model
        for (x, y) in train_loader
            x = x |> dev
            y = y |> dev
            (loss, y_pred, st), back = pullback(compute_loss, x, y, model, ps, st)
            gs = back((one(loss), nothing, nothing))[4]
            opt_state, ps = Optimisers.update(opt_state, ps, gs)

            println("Epoch [$epoch]: Loss $loss")
        end

        # Validate the model
        st_ = Lux.testmode(st)
        for (x, y) in val_loader
            x = x |> dev
            y = y |> dev
            (loss, y_pred, st_) = compute_loss(x, y, model, ps, st_)
            acc = accuracy(y_pred, y)
            println("Validation: Loss $loss Accuracy $acc")
        end
    end

    return (ps, st) |> cpu_device()
end

ps_trained, st_trained = main()
```


```
┌ Warning: `replicate` doesn't work for `TaskLocalRNG`. Returning the same `TaskLocalRNG`.
└ @ LuxCore ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/LuxCore/t4mG0/src/LuxCore.jl:13
Epoch [1]: Loss 0.56219864
Epoch [1]: Loss 0.51523095
Epoch [1]: Loss 0.48365018
Epoch [1]: Loss 0.4413105
Epoch [1]: Loss 0.42048597
Epoch [1]: Loss 0.4013239
Epoch [1]: Loss 0.3953247
Validation: Loss 0.3678489 Accuracy 1.0
Validation: Loss 0.3649088 Accuracy 1.0
Epoch [2]: Loss 0.37118214
Epoch [2]: Loss 0.34694228
Epoch [2]: Loss 0.3337921
Epoch [2]: Loss 0.31843352
Epoch [2]: Loss 0.30584523
Epoch [2]: Loss 0.2829361
Epoch [2]: Loss 0.27180228
Validation: Loss 0.2579401 Accuracy 1.0
Validation: Loss 0.2559074 Accuracy 1.0
Epoch [3]: Loss 0.26100865
Epoch [3]: Loss 0.2440408
Epoch [3]: Loss 0.23034632
Epoch [3]: Loss 0.22184965
Epoch [3]: Loss 0.20974205
Epoch [3]: Loss 0.19904274
Epoch [3]: Loss 0.18626328
Validation: Loss 0.17989004 Accuracy 1.0
Validation: Loss 0.17867896 Accuracy 1.0
Epoch [4]: Loss 0.18013218
Epoch [4]: Loss 0.17287448
Epoch [4]: Loss 0.162247
Epoch [4]: Loss 0.15325756
Epoch [4]: Loss 0.14854324
Epoch [4]: Loss 0.13996866
Epoch [4]: Loss 0.13327268
Validation: Loss 0.12807763 Accuracy 1.0
Validation: Loss 0.12723117 Accuracy 1.0
Epoch [5]: Loss 0.12846386
Epoch [5]: Loss 0.121599264
Epoch [5]: Loss 0.116471805
Epoch [5]: Loss 0.11139057
Epoch [5]: Loss 0.10651459
Epoch [5]: Loss 0.100890785
Epoch [5]: Loss 0.09823166
Validation: Loss 0.092880696 Accuracy 1.0
Validation: Loss 0.09213849 Accuracy 1.0
Epoch [6]: Loss 0.0941779
Epoch [6]: Loss 0.0860093
Epoch [6]: Loss 0.08584021
Epoch [6]: Loss 0.081216685
Epoch [6]: Loss 0.07768339
Epoch [6]: Loss 0.07463075
Epoch [6]: Loss 0.073942244
Validation: Loss 0.06837754 Accuracy 1.0
Validation: Loss 0.0676964 Accuracy 1.0
Epoch [7]: Loss 0.06879418
Epoch [7]: Loss 0.065937534
Epoch [7]: Loss 0.06327872
Epoch [7]: Loss 0.059918378
Epoch [7]: Loss 0.05718837
Epoch [7]: Loss 0.055784576
Epoch [7]: Loss 0.050523926
Validation: Loss 0.050939858 Accuracy 1.0
Validation: Loss 0.050337143 Accuracy 1.0
Epoch [8]: Loss 0.05093839
Epoch [8]: Loss 0.047925606
Epoch [8]: Loss 0.047354046
Epoch [8]: Loss 0.045574896
Epoch [8]: Loss 0.04329975
Epoch [8]: Loss 0.041412808
Epoch [8]: Loss 0.041926336
Validation: Loss 0.03829498 Accuracy 1.0
Validation: Loss 0.037741534 Accuracy 1.0
Epoch [9]: Loss 0.03839671
Epoch [9]: Loss 0.037378617
Epoch [9]: Loss 0.034957904
Epoch [9]: Loss 0.032996707
Epoch [9]: Loss 0.032925695
Epoch [9]: Loss 0.032852575
Epoch [9]: Loss 0.030293114
Validation: Loss 0.02924452 Accuracy 1.0
Validation: Loss 0.02874907 Accuracy 1.0
Epoch [10]: Loss 0.02883349
Epoch [10]: Loss 0.027803633
Epoch [10]: Loss 0.027415216
Epoch [10]: Loss 0.026352197
Epoch [10]: Loss 0.026261715
Epoch [10]: Loss 0.024494428
Epoch [10]: Loss 0.025626883
Validation: Loss 0.023004755 Accuracy 1.0
Validation: Loss 0.02256384 Accuracy 1.0
Epoch [11]: Loss 0.023098934
Epoch [11]: Loss 0.022349305
Epoch [11]: Loss 0.022333585
Epoch [11]: Loss 0.020741325
Epoch [11]: Loss 0.020575901
Epoch [11]: Loss 0.019423442
Epoch [11]: Loss 0.020253152
Validation: Loss 0.018718753 Accuracy 1.0
Validation: Loss 0.018336488 Accuracy 1.0
Epoch [12]: Loss 0.019848824
Epoch [12]: Loss 0.01736341
Epoch [12]: Loss 0.018369384
Epoch [12]: Loss 0.017250374
Epoch [12]: Loss 0.017015025
Epoch [12]: Loss 0.016618198
Epoch [12]: Loss 0.01444486
Validation: Loss 0.015730876 Accuracy 1.0
Validation: Loss 0.015393542 Accuracy 1.0
Epoch [13]: Loss 0.01607993
Epoch [13]: Loss 0.015427083
Epoch [13]: Loss 0.015198043
Epoch [13]: Loss 0.014983408
Epoch [13]: Loss 0.014818013
Epoch [13]: Loss 0.013724842
Epoch [13]: Loss 0.013079181
Validation: Loss 0.013597451 Accuracy 1.0
Validation: Loss 0.013296329 Accuracy 1.0
Epoch [14]: Loss 0.013629274
Epoch [14]: Loss 0.0129314065
Epoch [14]: Loss 0.013220712
Epoch [14]: Loss 0.013585413
Epoch [14]: Loss 0.01290969
Epoch [14]: Loss 0.01202965
Epoch [14]: Loss 0.012547791
Validation: Loss 0.0120011065 Accuracy 1.0
Validation: Loss 0.011728939 Accuracy 1.0
Epoch [15]: Loss 0.01290128
Epoch [15]: Loss 0.012082396
Epoch [15]: Loss 0.011243814
Epoch [15]: Loss 0.010808473
Epoch [15]: Loss 0.011369547
Epoch [15]: Loss 0.011040544
Epoch [15]: Loss 0.011512194
Validation: Loss 0.010748232 Accuracy 1.0
Validation: Loss 0.01049844 Accuracy 1.0
Epoch [16]: Loss 0.011214033
Epoch [16]: Loss 0.010413749
Epoch [16]: Loss 0.010158308
Epoch [16]: Loss 0.010689618
Epoch [16]: Loss 0.009780605
Epoch [16]: Loss 0.010150408
Epoch [16]: Loss 0.01047505
Validation: Loss 0.009729087 Accuracy 1.0
Validation: Loss 0.009503027 Accuracy 1.0
Epoch [17]: Loss 0.0091936
Epoch [17]: Loss 0.0098106135
Epoch [17]: Loss 0.009553109
Epoch [17]: Loss 0.009287022
Epoch [17]: Loss 0.009652537
Epoch [17]: Loss 0.009351825
Epoch [17]: Loss 0.008724324
Validation: Loss 0.008882927 Accuracy 1.0
Validation: Loss 0.008670893 Accuracy 1.0
Epoch [18]: Loss 0.008757358
Epoch [18]: Loss 0.008496236
Epoch [18]: Loss 0.009119667
Epoch [18]: Loss 0.008903662
Epoch [18]: Loss 0.008318211
Epoch [18]: Loss 0.008322838
Epoch [18]: Loss 0.008528218
Validation: Loss 0.008165628 Accuracy 1.0
Validation: Loss 0.0079724705 Accuracy 1.0
Epoch [19]: Loss 0.008719694
Epoch [19]: Loss 0.007801553
Epoch [19]: Loss 0.0074100634
Epoch [19]: Loss 0.008027736
Epoch [19]: Loss 0.0081958985
Epoch [19]: Loss 0.007675791
Epoch [19]: Loss 0.007880426
Validation: Loss 0.0075452626 Accuracy 1.0
Validation: Loss 0.0073604425 Accuracy 1.0
Epoch [20]: Loss 0.007663911
Epoch [20]: Loss 0.0075449822
Epoch [20]: Loss 0.007362173
Epoch [20]: Loss 0.007576653
Epoch [20]: Loss 0.0072928127
Epoch [20]: Loss 0.0068943882
Epoch [20]: Loss 0.0070477966
Validation: Loss 0.0070031285 Accuracy 1.0
Validation: Loss 0.0068328874 Accuracy 1.0
Epoch [21]: Loss 0.0066581434
Epoch [21]: Loss 0.0069510243
Epoch [21]: Loss 0.0064102774
Epoch [21]: Loss 0.006961026
Epoch [21]: Loss 0.007180891
Epoch [21]: Loss 0.0068694353
Epoch [21]: Loss 0.007224362
Validation: Loss 0.0065299324 Accuracy 1.0
Validation: Loss 0.0063681803 Accuracy 1.0
Epoch [22]: Loss 0.00661352
Epoch [22]: Loss 0.0058480552
Epoch [22]: Loss 0.006838514
Epoch [22]: Loss 0.0062865233
Epoch [22]: Loss 0.0064648665
Epoch [22]: Loss 0.0066399956
Epoch [22]: Loss 0.005240106
Validation: Loss 0.0061078873 Accuracy 1.0
Validation: Loss 0.0059559215 Accuracy 1.0
Epoch [23]: Loss 0.006357578
Epoch [23]: Loss 0.0060940203
Epoch [23]: Loss 0.006036909
Epoch [23]: Loss 0.0060478086
Epoch [23]: Loss 0.0055826763
Epoch [23]: Loss 0.005771838
Epoch [23]: Loss 0.0063576195
Validation: Loss 0.0057331296 Accuracy 1.0
Validation: Loss 0.0055890568 Accuracy 1.0
Epoch [24]: Loss 0.005832684
Epoch [24]: Loss 0.0057121236
Epoch [24]: Loss 0.0060292613
Epoch [24]: Loss 0.0056361873
Epoch [24]: Loss 0.005071752
Epoch [24]: Loss 0.005655307
Epoch [24]: Loss 0.0051114196
Validation: Loss 0.005394958 Accuracy 1.0
Validation: Loss 0.005258644 Accuracy 1.0
Epoch [25]: Loss 0.0053705215
Epoch [25]: Loss 0.0055283494
Epoch [25]: Loss 0.005261168
Epoch [25]: Loss 0.0052237324
Epoch [25]: Loss 0.005355524
Epoch [25]: Loss 0.0051739216
Epoch [25]: Loss 0.005018285
Validation: Loss 0.0050910683 Accuracy 1.0
Validation: Loss 0.0049624494 Accuracy 1.0

```


<a id='Saving-the-Model'></a>

## Saving the Model


We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don't save the model


```julia
@save "trained_model.jld2" {compress = true} ps_trained st_trained
```


Let's try loading the model


```julia
@load "trained_model.jld2" ps_trained st_trained
```


```
2-element Vector{Symbol}:
 :ps_trained
 :st_trained
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
  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.095 GiB / 4.750 GiB available)
┌ Warning: LuxAMDGPU is loaded but the AMDGPU is not functional.
└ @ LuxAMDGPU ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/LuxAMDGPU/sGa0S/src/LuxAMDGPU.jl:19

```


---


*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

