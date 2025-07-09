


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
Epoch [1]: Loss 0.56122816
Epoch [1]: Loss 0.5132421
Epoch [1]: Loss 0.47390133
Epoch [1]: Loss 0.44738066
Epoch [1]: Loss 0.4157251
Epoch [1]: Loss 0.40004316
Epoch [1]: Loss 0.38581663
Validation: Loss 0.38079154 Accuracy 1.0
Validation: Loss 0.36209294 Accuracy 1.0
Epoch [2]: Loss 0.3721903
Epoch [2]: Loss 0.35231066
Epoch [2]: Loss 0.3261134
Epoch [2]: Loss 0.30861035
Epoch [2]: Loss 0.2957699
Epoch [2]: Loss 0.2844334
Epoch [2]: Loss 0.2798446
Validation: Loss 0.26566783 Accuracy 1.0
Validation: Loss 0.2532907 Accuracy 1.0
Epoch [3]: Loss 0.25206405
Epoch [3]: Loss 0.2391364
Epoch [3]: Loss 0.2313126
Epoch [3]: Loss 0.22095928
Epoch [3]: Loss 0.20649536
Epoch [3]: Loss 0.19905809
Epoch [3]: Loss 0.1906898
Validation: Loss 0.18341061 Accuracy 1.0
Validation: Loss 0.17653722 Accuracy 1.0
Epoch [4]: Loss 0.17653653
Epoch [4]: Loss 0.17096186
Epoch [4]: Loss 0.16022289
Epoch [4]: Loss 0.15492734
Epoch [4]: Loss 0.14499216
Epoch [4]: Loss 0.13780512
Epoch [4]: Loss 0.13378745
Validation: Loss 0.12967902 Accuracy 1.0
Validation: Loss 0.12538734 Accuracy 1.0
Epoch [5]: Loss 0.12531984
Epoch [5]: Loss 0.12028654
Epoch [5]: Loss 0.116468295
Epoch [5]: Loss 0.10994354
Epoch [5]: Loss 0.10496262
Epoch [5]: Loss 0.09872067
Epoch [5]: Loss 0.09689447
Validation: Loss 0.09439968 Accuracy 1.0
Validation: Loss 0.09058539 Accuracy 1.0
Epoch [6]: Loss 0.090845406
Epoch [6]: Loss 0.087021895
Epoch [6]: Loss 0.0850915
Epoch [6]: Loss 0.079987705
Epoch [6]: Loss 0.07582793
Epoch [6]: Loss 0.07388522
Epoch [6]: Loss 0.0671367
Validation: Loss 0.07019704 Accuracy 1.0
Validation: Loss 0.066382706 Accuracy 1.0
Epoch [7]: Loss 0.0661852
Epoch [7]: Loss 0.06700981
Epoch [7]: Loss 0.061153278
Epoch [7]: Loss 0.05914691
Epoch [7]: Loss 0.056225665
Epoch [7]: Loss 0.053125713
Epoch [7]: Loss 0.05408698
Validation: Loss 0.052999817 Accuracy 1.0
Validation: Loss 0.04924028 Accuracy 1.0
Epoch [8]: Loss 0.048703257
Epoch [8]: Loss 0.04869066
Epoch [8]: Loss 0.04736167
Epoch [8]: Loss 0.045466334
Epoch [8]: Loss 0.042213455
Epoch [8]: Loss 0.04006204
Epoch [8]: Loss 0.035239264
Validation: Loss 0.04041663 Accuracy 1.0
Validation: Loss 0.036896452 Accuracy 1.0
Epoch [9]: Loss 0.036154103
Epoch [9]: Loss 0.035626397
Epoch [9]: Loss 0.034266792
Epoch [9]: Loss 0.035747357
Epoch [9]: Loss 0.031991985
Epoch [9]: Loss 0.03115772
Epoch [9]: Loss 0.030908493
Validation: Loss 0.03142257 Accuracy 1.0
Validation: Loss 0.028126307 Accuracy 1.0
Epoch [10]: Loss 0.029462555
Epoch [10]: Loss 0.029337067
Epoch [10]: Loss 0.026928972
Epoch [10]: Loss 0.02555028
Epoch [10]: Loss 0.023555331
Epoch [10]: Loss 0.023863953
Epoch [10]: Loss 0.023108687
Validation: Loss 0.024972335 Accuracy 1.0
Validation: Loss 0.022087447 Accuracy 1.0
Epoch [11]: Loss 0.022820171
Epoch [11]: Loss 0.021899063
Epoch [11]: Loss 0.021092372
Epoch [11]: Loss 0.021375122
Epoch [11]: Loss 0.019179031
Epoch [11]: Loss 0.020029482
Epoch [11]: Loss 0.018679785
Validation: Loss 0.02046242 Accuracy 1.0
Validation: Loss 0.017980121 Accuracy 1.0
Epoch [12]: Loss 0.0179513
Epoch [12]: Loss 0.018529087
Epoch [12]: Loss 0.017143168
Epoch [12]: Loss 0.017734094
Epoch [12]: Loss 0.016449241
Epoch [12]: Loss 0.016624806
Epoch [12]: Loss 0.014741253
Validation: Loss 0.017257351 Accuracy 1.0
Validation: Loss 0.015125398 Accuracy 1.0
Epoch [13]: Loss 0.015605297
Epoch [13]: Loss 0.015531963
Epoch [13]: Loss 0.01531181
Epoch [13]: Loss 0.0147368405
Epoch [13]: Loss 0.013043068
Epoch [13]: Loss 0.014163591
Epoch [13]: Loss 0.014236214
Validation: Loss 0.014926745 Accuracy 1.0
Validation: Loss 0.013062466 Accuracy 1.0
Epoch [14]: Loss 0.013506635
Epoch [14]: Loss 0.012896566
Epoch [14]: Loss 0.013903799
Epoch [14]: Loss 0.012975112
Epoch [14]: Loss 0.011923848
Epoch [14]: Loss 0.012282634
Epoch [14]: Loss 0.010550659
Validation: Loss 0.013163439 Accuracy 1.0
Validation: Loss 0.011519487 Accuracy 1.0
Epoch [15]: Loss 0.011741195
Epoch [15]: Loss 0.011737939
Epoch [15]: Loss 0.011679541
Epoch [15]: Loss 0.011125841
Epoch [15]: Loss 0.011176439
Epoch [15]: Loss 0.01079707
Epoch [15]: Loss 0.011239901
Validation: Loss 0.011803429 Accuracy 1.0
Validation: Loss 0.010317664 Accuracy 1.0
Epoch [16]: Loss 0.009838069
Epoch [16]: Loss 0.010819303
Epoch [16]: Loss 0.010454284
Epoch [16]: Loss 0.009997958
Epoch [16]: Loss 0.01052962
Epoch [16]: Loss 0.009669703
Epoch [16]: Loss 0.01046617
Validation: Loss 0.010702874 Accuracy 1.0
Validation: Loss 0.009341997 Accuracy 1.0
Epoch [17]: Loss 0.008907326
Epoch [17]: Loss 0.009908378
Epoch [17]: Loss 0.009543471
Epoch [17]: Loss 0.009439357
Epoch [17]: Loss 0.009044854
Epoch [17]: Loss 0.008989626
Epoch [17]: Loss 0.00903623
Validation: Loss 0.009770414 Accuracy 1.0
Validation: Loss 0.008526985 Accuracy 1.0
Epoch [18]: Loss 0.008256413
Epoch [18]: Loss 0.008833174
Epoch [18]: Loss 0.008865327
Epoch [18]: Loss 0.008154667
Epoch [18]: Loss 0.008505262
Epoch [18]: Loss 0.00808756
Epoch [18]: Loss 0.009725671
Validation: Loss 0.008984244 Accuracy 1.0
Validation: Loss 0.007832435 Accuracy 1.0
Epoch [19]: Loss 0.008008484
Epoch [19]: Loss 0.007930219
Epoch [19]: Loss 0.007876024
Epoch [19]: Loss 0.007986517
Epoch [19]: Loss 0.0074681286
Epoch [19]: Loss 0.0076417867
Epoch [19]: Loss 0.008084593
Validation: Loss 0.008299297 Accuracy 1.0
Validation: Loss 0.0072347173 Accuracy 1.0
Epoch [20]: Loss 0.007163735
Epoch [20]: Loss 0.007209964
Epoch [20]: Loss 0.00766572
Epoch [20]: Loss 0.0073459437
Epoch [20]: Loss 0.007277419
Epoch [20]: Loss 0.0069468156
Epoch [20]: Loss 0.006687188
Validation: Loss 0.007705426 Accuracy 1.0
Validation: Loss 0.006712441 Accuracy 1.0
Epoch [21]: Loss 0.0072048465
Epoch [21]: Loss 0.006797564
Epoch [21]: Loss 0.006761606
Epoch [21]: Loss 0.006399435
Epoch [21]: Loss 0.007011886
Epoch [21]: Loss 0.006369492
Epoch [21]: Loss 0.006243564
Validation: Loss 0.0071870126 Accuracy 1.0
Validation: Loss 0.0062572462 Accuracy 1.0
Epoch [22]: Loss 0.0064353123
Epoch [22]: Loss 0.0069045275
Epoch [22]: Loss 0.0069624046
Epoch [22]: Loss 0.0057010725
Epoch [22]: Loss 0.00601192
Epoch [22]: Loss 0.0059488807
Epoch [22]: Loss 0.0054498753
Validation: Loss 0.0067281425 Accuracy 1.0
Validation: Loss 0.0058526164 Accuracy 1.0
Epoch [23]: Loss 0.0058941683
Epoch [23]: Loss 0.0056786276
Epoch [23]: Loss 0.006017935
Epoch [23]: Loss 0.0060810046
Epoch [23]: Loss 0.0060943877
Epoch [23]: Loss 0.0057354895
Epoch [23]: Loss 0.005281723
Validation: Loss 0.0063246908 Accuracy 1.0
Validation: Loss 0.005495351 Accuracy 1.0
Epoch [24]: Loss 0.006127512
Epoch [24]: Loss 0.0058765914
Epoch [24]: Loss 0.005616461
Epoch [24]: Loss 0.0053911745
Epoch [24]: Loss 0.0050209756
Epoch [24]: Loss 0.0054028323
Epoch [24]: Loss 0.004822187
Validation: Loss 0.0059580086 Accuracy 1.0
Validation: Loss 0.0051735286 Accuracy 1.0
Epoch [25]: Loss 0.0051332912
Epoch [25]: Loss 0.005507709
Epoch [25]: Loss 0.005112962
Epoch [25]: Loss 0.0050922614
Epoch [25]: Loss 0.0053021377
Epoch [25]: Loss 0.004964512
Epoch [25]: Loss 0.00605146
Validation: Loss 0.0056295763 Accuracy 1.0
Validation: Loss 0.004884051 Accuracy 1.0

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
  JULIA_PROJECT = /var/lib/buildkite-agent/builds/gpuci-14/julialang/lux-dot-jl/docs/Project.toml
  JULIA_AMDGPU_LOGGING_ENABLED = true
  JULIA_DEBUG = Literate
  JULIA_CPU_THREADS = 2
  JULIA_NUM_THREADS = 48
  JULIA_LOAD_PATH = @:@v#.#:@stdlib
  JULIA_CUDA_HARD_MEMORY_LIMIT = 25%

CUDA runtime 12.3, artifact installation
CUDA driver 12.3
NVIDIA driver 545.23.8

CUDA libraries: 
- CUBLAS: 12.3.4
- CURAND: 10.3.4
- CUFFT: 11.0.12
- CUSOLVER: 11.5.4
- CUSPARSE: 12.2.0
- CUPTI: 21.0.0
- NVML: 12.0.0+545.23.8

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
  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.359 GiB / 4.750 GiB available)
┌ Warning: LuxAMDGPU is loaded but the AMDGPU is not functional.
└ @ LuxAMDGPU ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/LuxAMDGPU/sGa0S/src/LuxAMDGPU.jl:19

```


---


*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

