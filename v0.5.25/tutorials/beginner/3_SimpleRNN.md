


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
Epoch [1]: Loss 0.56371677
Epoch [1]: Loss 0.511444
Epoch [1]: Loss 0.47410044
Epoch [1]: Loss 0.44776702
Epoch [1]: Loss 0.41685304
Epoch [1]: Loss 0.40030515
Epoch [1]: Loss 0.3705929
Validation: Loss 0.3739136 Accuracy 1.0
Validation: Loss 0.3763235 Accuracy 1.0
Epoch [2]: Loss 0.36657873
Epoch [2]: Loss 0.34901023
Epoch [2]: Loss 0.33842215
Epoch [2]: Loss 0.3103344
Epoch [2]: Loss 0.29035038
Epoch [2]: Loss 0.286722
Epoch [2]: Loss 0.27443355
Validation: Loss 0.26185477 Accuracy 1.0
Validation: Loss 0.26417053 Accuracy 1.0
Epoch [3]: Loss 0.250595
Epoch [3]: Loss 0.24793768
Epoch [3]: Loss 0.23392135
Epoch [3]: Loss 0.22225207
Epoch [3]: Loss 0.20684987
Epoch [3]: Loss 0.19220671
Epoch [3]: Loss 0.19002713
Validation: Loss 0.1810775 Accuracy 1.0
Validation: Loss 0.18274799 Accuracy 1.0
Epoch [4]: Loss 0.17454857
Epoch [4]: Loss 0.16902441
Epoch [4]: Loss 0.15851262
Epoch [4]: Loss 0.15687552
Epoch [4]: Loss 0.14821076
Epoch [4]: Loss 0.14007978
Epoch [4]: Loss 0.12763885
Validation: Loss 0.12837385 Accuracy 1.0
Validation: Loss 0.12962815 Accuracy 1.0
Epoch [5]: Loss 0.12391113
Epoch [5]: Loss 0.122634396
Epoch [5]: Loss 0.11621217
Epoch [5]: Loss 0.10986562
Epoch [5]: Loss 0.10507065
Epoch [5]: Loss 0.09890335
Epoch [5]: Loss 0.09753703
Validation: Loss 0.093320936 Accuracy 1.0
Validation: Loss 0.09435924 Accuracy 1.0
Epoch [6]: Loss 0.09116772
Epoch [6]: Loss 0.085738316
Epoch [6]: Loss 0.08096885
Epoch [6]: Loss 0.081969514
Epoch [6]: Loss 0.07848319
Epoch [6]: Loss 0.07445756
Epoch [6]: Loss 0.07223338
Validation: Loss 0.069146626 Accuracy 1.0
Validation: Loss 0.070011266 Accuracy 1.0
Epoch [7]: Loss 0.066627175
Epoch [7]: Loss 0.06324355
Epoch [7]: Loss 0.06282787
Epoch [7]: Loss 0.05957313
Epoch [7]: Loss 0.05732356
Epoch [7]: Loss 0.055309076
Epoch [7]: Loss 0.05251997
Validation: Loss 0.051858403 Accuracy 1.0
Validation: Loss 0.05255384 Accuracy 1.0
Epoch [8]: Loss 0.048801117
Epoch [8]: Loss 0.048480667
Epoch [8]: Loss 0.04659126
Epoch [8]: Loss 0.046424314
Epoch [8]: Loss 0.042330634
Epoch [8]: Loss 0.04119297
Epoch [8]: Loss 0.03588528
Validation: Loss 0.03930341 Accuracy 1.0
Validation: Loss 0.039850954 Accuracy 1.0
Epoch [9]: Loss 0.037508253
Epoch [9]: Loss 0.035943393
Epoch [9]: Loss 0.0361139
Epoch [9]: Loss 0.03436319
Epoch [9]: Loss 0.03286673
Epoch [9]: Loss 0.02974714
Epoch [9]: Loss 0.02964848
Validation: Loss 0.030316979 Accuracy 1.0
Validation: Loss 0.030762127 Accuracy 1.0
Epoch [10]: Loss 0.02879231
Epoch [10]: Loss 0.028560415
Epoch [10]: Loss 0.027167113
Epoch [10]: Loss 0.025196236
Epoch [10]: Loss 0.02545802
Epoch [10]: Loss 0.02421618
Epoch [10]: Loss 0.023304325
Validation: Loss 0.024028696 Accuracy 1.0
Validation: Loss 0.024395654 Accuracy 1.0
Epoch [11]: Loss 0.022921085
Epoch [11]: Loss 0.021652207
Epoch [11]: Loss 0.021102006
Epoch [11]: Loss 0.02011004
Epoch [11]: Loss 0.020738462
Epoch [11]: Loss 0.020330213
Epoch [11]: Loss 0.019333754
Validation: Loss 0.01966207 Accuracy 1.0
Validation: Loss 0.019983001 Accuracy 1.0
Epoch [12]: Loss 0.019217828
Epoch [12]: Loss 0.0184287
Epoch [12]: Loss 0.01753746
Epoch [12]: Loss 0.017656446
Epoch [12]: Loss 0.016302276
Epoch [12]: Loss 0.016107842
Epoch [12]: Loss 0.0139736235
Validation: Loss 0.016566915 Accuracy 1.0
Validation: Loss 0.016822238 Accuracy 1.0
Epoch [13]: Loss 0.015702978
Epoch [13]: Loss 0.014540378
Epoch [13]: Loss 0.015691463
Epoch [13]: Loss 0.014331127
Epoch [13]: Loss 0.014663866
Epoch [13]: Loss 0.013544618
Epoch [13]: Loss 0.015601161
Validation: Loss 0.014327088 Accuracy 1.0
Validation: Loss 0.014571899 Accuracy 1.0
Epoch [14]: Loss 0.01392062
Epoch [14]: Loss 0.01335464
Epoch [14]: Loss 0.013091152
Epoch [14]: Loss 0.012762748
Epoch [14]: Loss 0.012587881
Epoch [14]: Loss 0.011974597
Epoch [14]: Loss 0.01119967
Validation: Loss 0.0126285385 Accuracy 1.0
Validation: Loss 0.012836251 Accuracy 1.0
Epoch [15]: Loss 0.0123178605
Epoch [15]: Loss 0.011801089
Epoch [15]: Loss 0.012072599
Epoch [15]: Loss 0.010855473
Epoch [15]: Loss 0.01138348
Epoch [15]: Loss 0.00998167
Epoch [15]: Loss 0.011900126
Validation: Loss 0.011309456 Accuracy 1.0
Validation: Loss 0.011499692 Accuracy 1.0
Epoch [16]: Loss 0.011169883
Epoch [16]: Loss 0.01033945
Epoch [16]: Loss 0.0097381575
Epoch [16]: Loss 0.010145234
Epoch [16]: Loss 0.009989709
Epoch [16]: Loss 0.0100567965
Epoch [16]: Loss 0.010809666
Validation: Loss 0.010241656 Accuracy 1.0
Validation: Loss 0.010418712 Accuracy 1.0
Epoch [17]: Loss 0.009327468
Epoch [17]: Loss 0.009734323
Epoch [17]: Loss 0.008584525
Epoch [17]: Loss 0.009104807
Epoch [17]: Loss 0.009348607
Epoch [17]: Loss 0.009806592
Epoch [17]: Loss 0.009268496
Validation: Loss 0.009353134 Accuracy 1.0
Validation: Loss 0.009510893 Accuracy 1.0
Epoch [18]: Loss 0.008925518
Epoch [18]: Loss 0.008951698
Epoch [18]: Loss 0.00843966
Epoch [18]: Loss 0.008304184
Epoch [18]: Loss 0.00861743
Epoch [18]: Loss 0.00830004
Epoch [18]: Loss 0.0071531674
Validation: Loss 0.008596173 Accuracy 1.0
Validation: Loss 0.008745188 Accuracy 1.0
Epoch [19]: Loss 0.008206956
Epoch [19]: Loss 0.007711198
Epoch [19]: Loss 0.008365605
Epoch [19]: Loss 0.007699765
Epoch [19]: Loss 0.0075716246
Epoch [19]: Loss 0.00773654
Epoch [19]: Loss 0.0072414153
Validation: Loss 0.007952705 Accuracy 1.0
Validation: Loss 0.008088964 Accuracy 1.0
Epoch [20]: Loss 0.0074158437
Epoch [20]: Loss 0.0076126466
Epoch [20]: Loss 0.0073661255
Epoch [20]: Loss 0.0069544227
Epoch [20]: Loss 0.0075766193
Epoch [20]: Loss 0.0066084927
Epoch [20]: Loss 0.007774149
Validation: Loss 0.0073910477 Accuracy 1.0
Validation: Loss 0.0075202575 Accuracy 1.0
Epoch [21]: Loss 0.0066717127
Epoch [21]: Loss 0.0074326773
Epoch [21]: Loss 0.0066809664
Epoch [21]: Loss 0.006641823
Epoch [21]: Loss 0.0064884443
Epoch [21]: Loss 0.006916249
Epoch [21]: Loss 0.0058681956
Validation: Loss 0.006893435 Accuracy 1.0
Validation: Loss 0.007013739 Accuracy 1.0
Epoch [22]: Loss 0.0070244125
Epoch [22]: Loss 0.006306324
Epoch [22]: Loss 0.0063859373
Epoch [22]: Loss 0.0061884494
Epoch [22]: Loss 0.006189266
Epoch [22]: Loss 0.0060048075
Epoch [22]: Loss 0.0056366893
Validation: Loss 0.0064538056 Accuracy 1.0
Validation: Loss 0.006565489 Accuracy 1.0
Epoch [23]: Loss 0.006065852
Epoch [23]: Loss 0.0060522174
Epoch [23]: Loss 0.006313501
Epoch [23]: Loss 0.0057540173
Epoch [23]: Loss 0.005662987
Epoch [23]: Loss 0.0057507604
Epoch [23]: Loss 0.005654332
Validation: Loss 0.0060628103 Accuracy 1.0
Validation: Loss 0.0061705327 Accuracy 1.0
Epoch [24]: Loss 0.0058130883
Epoch [24]: Loss 0.0056798253
Epoch [24]: Loss 0.005940421
Epoch [24]: Loss 0.0052474537
Epoch [24]: Loss 0.0054572723
Epoch [24]: Loss 0.005289708
Epoch [24]: Loss 0.0054617785
Validation: Loss 0.005709597 Accuracy 1.0
Validation: Loss 0.0058096657 Accuracy 1.0
Epoch [25]: Loss 0.0054784045
Epoch [25]: Loss 0.005417645
Epoch [25]: Loss 0.0053004753
Epoch [25]: Loss 0.005098868
Epoch [25]: Loss 0.004979431
Epoch [25]: Loss 0.005178516
Epoch [25]: Loss 0.0053134505
Validation: Loss 0.00538979 Accuracy 1.0
Validation: Loss 0.005485205 Accuracy 1.0

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
  JULIA_PROJECT = /var/lib/buildkite-agent/builds/gpuci-5/julialang/lux-dot-jl/docs/Project.toml
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
  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.359 GiB / 4.750 GiB available)
┌ Warning: LuxAMDGPU is loaded but the AMDGPU is not functional.
└ @ LuxAMDGPU ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/LuxAMDGPU/sGa0S/src/LuxAMDGPU.jl:19

```


---


*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
