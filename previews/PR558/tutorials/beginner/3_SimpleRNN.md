


<a id='Training-a-Simple-LSTM'></a>

# Training a Simple LSTM


In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:


1. Create custom Lux models.
2. Become familiar with the Lux recurrent neural network API.
3. Training using Optimisers.jl and Zygote.jl.


<a id='Package-Imports'></a>

## Package Imports


```julia
using ADTypes, Lux, LuxAMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random,
      Statistics
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

function compute_loss(model, ps, st, (x, y))
    y_pred, st = model(x, ps, st)
    return binarycrossentropy(y_pred, y), st, (; y_pred=y_pred)
end

matches(y_pred, y_true) = sum((y_pred .> 0.5f0) .== y_true)
accuracy(y_pred, y_true) = matches(y_pred, y_true) / length(y_pred)
```


```
accuracy (generic function with 1 method)
```


<a id='Training-the-Model'></a>

## Training the Model


```julia
function main()
    # Get the dataloaders
    (train_loader, val_loader) = get_dataloaders()

    # Create the model
    model = SpiralClassifier(2, 8, 1)
    rng = Xoshiro(0)

    dev = gpu_device()
    train_state = Lux.Experimental.TrainState(
        rng, model, Adam(0.01f0); transform_variables=dev)

    for epoch in 1:25
        # Train the model
        for (x, y) in train_loader
            x = x |> dev
            y = y |> dev

            gs, loss, _, train_state = Lux.Experimental.compute_gradients(
                AutoZygote(), compute_loss, (x, y), train_state)
            train_state = Lux.Experimental.apply_gradients(train_state, gs)

            @printf "Epoch [%3d]: Loss %4.5f\n" epoch loss
        end

        # Validate the model
        st_ = Lux.testmode(train_state.states)
        for (x, y) in val_loader
            x = x |> dev
            y = y |> dev
            loss, st_, ret = compute_loss(model, train_state.parameters, st_, (x, y))
            acc = accuracy(ret.y_pred, y)
            @printf "Validation: Loss %4.5f Accuracy %4.5f\n" loss acc
        end
    end

    return (train_state.parameters, train_state.states) |> cpu_device()
end

ps_trained, st_trained = main()
```


```
Epoch [  1]: Loss 0.56244
Epoch [  1]: Loss 0.50852
Epoch [  1]: Loss 0.47945
Epoch [  1]: Loss 0.45007
Epoch [  1]: Loss 0.43151
Epoch [  1]: Loss 0.40202
Epoch [  1]: Loss 0.38494
Validation: Loss 0.36962 Accuracy 1.00000
Validation: Loss 0.38194 Accuracy 1.00000
Epoch [  2]: Loss 0.36173
Epoch [  2]: Loss 0.35453
Epoch [  2]: Loss 0.33613
Epoch [  2]: Loss 0.31867
Epoch [  2]: Loss 0.30801
Epoch [  2]: Loss 0.28206
Epoch [  2]: Loss 0.27307
Validation: Loss 0.25840 Accuracy 1.00000
Validation: Loss 0.26650 Accuracy 1.00000
Epoch [  3]: Loss 0.25885
Epoch [  3]: Loss 0.24424
Epoch [  3]: Loss 0.23390
Epoch [  3]: Loss 0.22049
Epoch [  3]: Loss 0.21048
Epoch [  3]: Loss 0.19994
Epoch [  3]: Loss 0.18893
Validation: Loss 0.17983 Accuracy 1.00000
Validation: Loss 0.18430 Accuracy 1.00000
Epoch [  4]: Loss 0.17997
Epoch [  4]: Loss 0.17285
Epoch [  4]: Loss 0.16633
Epoch [  4]: Loss 0.15429
Epoch [  4]: Loss 0.14594
Epoch [  4]: Loss 0.13944
Epoch [  4]: Loss 0.13851
Validation: Loss 0.12798 Accuracy 1.00000
Validation: Loss 0.13082 Accuracy 1.00000
Epoch [  5]: Loss 0.12860
Epoch [  5]: Loss 0.11985
Epoch [  5]: Loss 0.11594
Epoch [  5]: Loss 0.11338
Epoch [  5]: Loss 0.10766
Epoch [  5]: Loss 0.10245
Epoch [  5]: Loss 0.09704
Validation: Loss 0.09294 Accuracy 1.00000
Validation: Loss 0.09542 Accuracy 1.00000
Epoch [  6]: Loss 0.09227
Epoch [  6]: Loss 0.08746
Epoch [  6]: Loss 0.08514
Epoch [  6]: Loss 0.08413
Epoch [  6]: Loss 0.07650
Epoch [  6]: Loss 0.07598
Epoch [  6]: Loss 0.07358
Validation: Loss 0.06856 Accuracy 1.00000
Validation: Loss 0.07094 Accuracy 1.00000
Epoch [  7]: Loss 0.06801
Epoch [  7]: Loss 0.06663
Epoch [  7]: Loss 0.06253
Epoch [  7]: Loss 0.06163
Epoch [  7]: Loss 0.05852
Epoch [  7]: Loss 0.05383
Epoch [  7]: Loss 0.05458
Validation: Loss 0.05117 Accuracy 1.00000
Validation: Loss 0.05334 Accuracy 1.00000
Epoch [  8]: Loss 0.05126
Epoch [  8]: Loss 0.04981
Epoch [  8]: Loss 0.04821
Epoch [  8]: Loss 0.04549
Epoch [  8]: Loss 0.04369
Epoch [  8]: Loss 0.04008
Epoch [  8]: Loss 0.03678
Validation: Loss 0.03851 Accuracy 1.00000
Validation: Loss 0.04046 Accuracy 1.00000
Epoch [  9]: Loss 0.03723
Epoch [  9]: Loss 0.03705
Epoch [  9]: Loss 0.03669
Epoch [  9]: Loss 0.03459
Epoch [  9]: Loss 0.03234
Epoch [  9]: Loss 0.03119
Epoch [  9]: Loss 0.03192
Validation: Loss 0.02943 Accuracy 1.00000
Validation: Loss 0.03124 Accuracy 1.00000
Epoch [ 10]: Loss 0.02726
Epoch [ 10]: Loss 0.02939
Epoch [ 10]: Loss 0.02759
Epoch [ 10]: Loss 0.02572
Epoch [ 10]: Loss 0.02556
Epoch [ 10]: Loss 0.02561
Epoch [ 10]: Loss 0.02408
Validation: Loss 0.02314 Accuracy 1.00000
Validation: Loss 0.02476 Accuracy 1.00000
Epoch [ 11]: Loss 0.02362
Epoch [ 11]: Loss 0.02185
Epoch [ 11]: Loss 0.02132
Epoch [ 11]: Loss 0.02219
Epoch [ 11]: Loss 0.02047
Epoch [ 11]: Loss 0.01901
Epoch [ 11]: Loss 0.01821
Validation: Loss 0.01882 Accuracy 1.00000
Validation: Loss 0.02021 Accuracy 1.00000
Epoch [ 12]: Loss 0.01875
Epoch [ 12]: Loss 0.01874
Epoch [ 12]: Loss 0.01789
Epoch [ 12]: Loss 0.01762
Epoch [ 12]: Loss 0.01696
Epoch [ 12]: Loss 0.01544
Epoch [ 12]: Loss 0.01620
Validation: Loss 0.01580 Accuracy 1.00000
Validation: Loss 0.01701 Accuracy 1.00000
Epoch [ 13]: Loss 0.01493
Epoch [ 13]: Loss 0.01521
Epoch [ 13]: Loss 0.01472
Epoch [ 13]: Loss 0.01523
Epoch [ 13]: Loss 0.01507
Epoch [ 13]: Loss 0.01401
Epoch [ 13]: Loss 0.01431
Validation: Loss 0.01363 Accuracy 1.00000
Validation: Loss 0.01470 Accuracy 1.00000
Epoch [ 14]: Loss 0.01337
Epoch [ 14]: Loss 0.01316
Epoch [ 14]: Loss 0.01343
Epoch [ 14]: Loss 0.01296
Epoch [ 14]: Loss 0.01286
Epoch [ 14]: Loss 0.01201
Epoch [ 14]: Loss 0.01181
Validation: Loss 0.01201 Accuracy 1.00000
Validation: Loss 0.01296 Accuracy 1.00000
Epoch [ 15]: Loss 0.01184
Epoch [ 15]: Loss 0.01165
Epoch [ 15]: Loss 0.01205
Epoch [ 15]: Loss 0.01137
Epoch [ 15]: Loss 0.01113
Epoch [ 15]: Loss 0.01056
Epoch [ 15]: Loss 0.01182
Validation: Loss 0.01074 Accuracy 1.00000
Validation: Loss 0.01159 Accuracy 1.00000
Epoch [ 16]: Loss 0.00940
Epoch [ 16]: Loss 0.01104
Epoch [ 16]: Loss 0.01063
Epoch [ 16]: Loss 0.00970
Epoch [ 16]: Loss 0.01066
Epoch [ 16]: Loss 0.01069
Epoch [ 16]: Loss 0.00844
Validation: Loss 0.00972 Accuracy 1.00000
Validation: Loss 0.01050 Accuracy 1.00000
Epoch [ 17]: Loss 0.00986
Epoch [ 17]: Loss 0.00997
Epoch [ 17]: Loss 0.00945
Epoch [ 17]: Loss 0.00951
Epoch [ 17]: Loss 0.00859
Epoch [ 17]: Loss 0.00878
Epoch [ 17]: Loss 0.00898
Validation: Loss 0.00887 Accuracy 1.00000
Validation: Loss 0.00959 Accuracy 1.00000
Epoch [ 18]: Loss 0.00900
Epoch [ 18]: Loss 0.00835
Epoch [ 18]: Loss 0.00859
Epoch [ 18]: Loss 0.00808
Epoch [ 18]: Loss 0.00921
Epoch [ 18]: Loss 0.00815
Epoch [ 18]: Loss 0.00806
Validation: Loss 0.00815 Accuracy 1.00000
Validation: Loss 0.00882 Accuracy 1.00000
Epoch [ 19]: Loss 0.00839
Epoch [ 19]: Loss 0.00836
Epoch [ 19]: Loss 0.00751
Epoch [ 19]: Loss 0.00759
Epoch [ 19]: Loss 0.00803
Epoch [ 19]: Loss 0.00738
Epoch [ 19]: Loss 0.00767
Validation: Loss 0.00753 Accuracy 1.00000
Validation: Loss 0.00815 Accuracy 1.00000
Epoch [ 20]: Loss 0.00771
Epoch [ 20]: Loss 0.00675
Epoch [ 20]: Loss 0.00701
Epoch [ 20]: Loss 0.00724
Epoch [ 20]: Loss 0.00728
Epoch [ 20]: Loss 0.00763
Epoch [ 20]: Loss 0.00753
Validation: Loss 0.00699 Accuracy 1.00000
Validation: Loss 0.00758 Accuracy 1.00000
Epoch [ 21]: Loss 0.00686
Epoch [ 21]: Loss 0.00694
Epoch [ 21]: Loss 0.00685
Epoch [ 21]: Loss 0.00720
Epoch [ 21]: Loss 0.00637
Epoch [ 21]: Loss 0.00637
Epoch [ 21]: Loss 0.00696
Validation: Loss 0.00651 Accuracy 1.00000
Validation: Loss 0.00706 Accuracy 1.00000
Epoch [ 22]: Loss 0.00629
Epoch [ 22]: Loss 0.00661
Epoch [ 22]: Loss 0.00609
Epoch [ 22]: Loss 0.00611
Epoch [ 22]: Loss 0.00643
Epoch [ 22]: Loss 0.00636
Epoch [ 22]: Loss 0.00635
Validation: Loss 0.00609 Accuracy 1.00000
Validation: Loss 0.00660 Accuracy 1.00000
Epoch [ 23]: Loss 0.00611
Epoch [ 23]: Loss 0.00583
Epoch [ 23]: Loss 0.00622
Epoch [ 23]: Loss 0.00617
Epoch [ 23]: Loss 0.00560
Epoch [ 23]: Loss 0.00573
Epoch [ 23]: Loss 0.00528
Validation: Loss 0.00571 Accuracy 1.00000
Validation: Loss 0.00620 Accuracy 1.00000
Epoch [ 24]: Loss 0.00588
Epoch [ 24]: Loss 0.00561
Epoch [ 24]: Loss 0.00572
Epoch [ 24]: Loss 0.00544
Epoch [ 24]: Loss 0.00527
Epoch [ 24]: Loss 0.00561
Epoch [ 24]: Loss 0.00476
Validation: Loss 0.00537 Accuracy 1.00000
Validation: Loss 0.00583 Accuracy 1.00000
Epoch [ 25]: Loss 0.00543
Epoch [ 25]: Loss 0.00528
Epoch [ 25]: Loss 0.00538
Epoch [ 25]: Loss 0.00574
Epoch [ 25]: Loss 0.00493
Epoch [ 25]: Loss 0.00493
Epoch [ 25]: Loss 0.00418
Validation: Loss 0.00507 Accuracy 1.00000
Validation: Loss 0.00551 Accuracy 1.00000

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
  JULIA_PROJECT = /var/lib/buildkite-agent/builds/gpuci-7/julialang/lux-dot-jl/docs/Project.toml
  JULIA_AMDGPU_LOGGING_ENABLED = true
  JULIA_DEBUG = Literate
  JULIA_CPU_THREADS = 2
  JULIA_NUM_THREADS = 48
  JULIA_LOAD_PATH = @:@v#.#:@stdlib
  JULIA_CUDA_HARD_MEMORY_LIMIT = 25%

CUDA runtime 12.3, artifact installation
CUDA driver 12.4
NVIDIA driver 550.54.15

CUDA libraries: 
- CUBLAS: 12.3.4
- CURAND: 10.3.4
- CUFFT: 11.0.12
- CUSOLVER: 11.5.4
- CUSPARSE: 12.2.0
- CUPTI: 21.0.0
- NVML: 12.0.0+550.54.15

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
  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.328 GiB / 4.750 GiB available)
┌ Warning: LuxAMDGPU is loaded but the AMDGPU is not functional.
└ @ LuxAMDGPU ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/LuxAMDGPU/sGa0S/src/LuxAMDGPU.jl:19

```


---


*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

