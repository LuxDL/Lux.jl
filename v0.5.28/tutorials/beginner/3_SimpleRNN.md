


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
Epoch [1]: Loss 0.562466
Epoch [1]: Loss 0.51045793
Epoch [1]: Loss 0.47355878
Epoch [1]: Loss 0.44110897
Epoch [1]: Loss 0.42466038
Epoch [1]: Loss 0.4047184
Epoch [1]: Loss 0.39617693
Validation: Loss 0.36246547 Accuracy 1.0
Validation: Loss 0.36631733 Accuracy 1.0
Epoch [2]: Loss 0.36646172
Epoch [2]: Loss 0.3511799
Epoch [2]: Loss 0.3338619
Epoch [2]: Loss 0.31603995
Epoch [2]: Loss 0.300273
Epoch [2]: Loss 0.2822481
Epoch [2]: Loss 0.27031958
Validation: Loss 0.25388724 Accuracy 1.0
Validation: Loss 0.25595438 Accuracy 1.0
Epoch [3]: Loss 0.25291392
Epoch [3]: Loss 0.24516988
Epoch [3]: Loss 0.23163792
Epoch [3]: Loss 0.22147745
Epoch [3]: Loss 0.20900458
Epoch [3]: Loss 0.19562145
Epoch [3]: Loss 0.18734963
Validation: Loss 0.17700619 Accuracy 1.0
Validation: Loss 0.17795037 Accuracy 1.0
Epoch [4]: Loss 0.17861225
Epoch [4]: Loss 0.1683875
Epoch [4]: Loss 0.1621432
Epoch [4]: Loss 0.1532538
Epoch [4]: Loss 0.1468495
Epoch [4]: Loss 0.1392852
Epoch [4]: Loss 0.13166308
Validation: Loss 0.12573063 Accuracy 1.0
Validation: Loss 0.12633653 Accuracy 1.0
Epoch [5]: Loss 0.1264346
Epoch [5]: Loss 0.119981945
Epoch [5]: Loss 0.11547021
Epoch [5]: Loss 0.10910138
Epoch [5]: Loss 0.106162444
Epoch [5]: Loss 0.10147768
Epoch [5]: Loss 0.09440764
Validation: Loss 0.09075983 Accuracy 1.0
Validation: Loss 0.09142318 Accuracy 1.0
Epoch [6]: Loss 0.09075469
Epoch [6]: Loss 0.086964615
Epoch [6]: Loss 0.08438568
Epoch [6]: Loss 0.081507444
Epoch [6]: Loss 0.076327085
Epoch [6]: Loss 0.073543526
Epoch [6]: Loss 0.0718967
Validation: Loss 0.06652105 Accuracy 1.0
Validation: Loss 0.06724769 Accuracy 1.0
Epoch [7]: Loss 0.06700897
Epoch [7]: Loss 0.06385116
Epoch [7]: Loss 0.06141842
Epoch [7]: Loss 0.060149528
Epoch [7]: Loss 0.05735492
Epoch [7]: Loss 0.055380087
Epoch [7]: Loss 0.05212902
Validation: Loss 0.049362414 Accuracy 1.0
Validation: Loss 0.050075438 Accuracy 1.0
Epoch [8]: Loss 0.05101807
Epoch [8]: Loss 0.04812645
Epoch [8]: Loss 0.04615236
Epoch [8]: Loss 0.04390265
Epoch [8]: Loss 0.04271665
Epoch [8]: Loss 0.041349344
Epoch [8]: Loss 0.037815675
Validation: Loss 0.036973193 Accuracy 1.0
Validation: Loss 0.037652895 Accuracy 1.0
Epoch [9]: Loss 0.037154898
Epoch [9]: Loss 0.035715982
Epoch [9]: Loss 0.03602562
Epoch [9]: Loss 0.03384151
Epoch [9]: Loss 0.032913856
Epoch [9]: Loss 0.030763606
Epoch [9]: Loss 0.02926112
Validation: Loss 0.028156858 Accuracy 1.0
Validation: Loss 0.028790308 Accuracy 1.0
Epoch [10]: Loss 0.028487187
Epoch [10]: Loss 0.026040208
Epoch [10]: Loss 0.028382968
Epoch [10]: Loss 0.026363233
Epoch [10]: Loss 0.024629924
Epoch [10]: Loss 0.02510789
Epoch [10]: Loss 0.024504885
Validation: Loss 0.022123072 Accuracy 1.0
Validation: Loss 0.022692502 Accuracy 1.0
Epoch [11]: Loss 0.022882113
Epoch [11]: Loss 0.023522124
Epoch [11]: Loss 0.022159223
Epoch [11]: Loss 0.019577662
Epoch [11]: Loss 0.019251589
Epoch [11]: Loss 0.019690309
Epoch [11]: Loss 0.019463226
Validation: Loss 0.018001357 Accuracy 1.0
Validation: Loss 0.018491216 Accuracy 1.0
Epoch [12]: Loss 0.017677968
Epoch [12]: Loss 0.01747987
Epoch [12]: Loss 0.018575808
Epoch [12]: Loss 0.017195934
Epoch [12]: Loss 0.016925525
Epoch [12]: Loss 0.01677715
Epoch [12]: Loss 0.016335623
Validation: Loss 0.01513888 Accuracy 1.0
Validation: Loss 0.015562764 Accuracy 1.0
Epoch [13]: Loss 0.015950989
Epoch [13]: Loss 0.0153554
Epoch [13]: Loss 0.015809257
Epoch [13]: Loss 0.0143444
Epoch [13]: Loss 0.014119103
Epoch [13]: Loss 0.013545454
Epoch [13]: Loss 0.013787945
Validation: Loss 0.013076011 Accuracy 1.0
Validation: Loss 0.013444344 Accuracy 1.0
Epoch [14]: Loss 0.013733086
Epoch [14]: Loss 0.0135611
Epoch [14]: Loss 0.012705318
Epoch [14]: Loss 0.01420505
Epoch [14]: Loss 0.011423074
Epoch [14]: Loss 0.012079604
Epoch [14]: Loss 0.011720823
Validation: Loss 0.011532082 Accuracy 1.0
Validation: Loss 0.011859651 Accuracy 1.0
Epoch [15]: Loss 0.011627066
Epoch [15]: Loss 0.012573771
Epoch [15]: Loss 0.011647998
Epoch [15]: Loss 0.010771445
Epoch [15]: Loss 0.011430441
Epoch [15]: Loss 0.011001726
Epoch [15]: Loss 0.009974889
Validation: Loss 0.01032858 Accuracy 1.0
Validation: Loss 0.010622444 Accuracy 1.0
Epoch [16]: Loss 0.010722721
Epoch [16]: Loss 0.011062598
Epoch [16]: Loss 0.010064267
Epoch [16]: Loss 0.009714608
Epoch [16]: Loss 0.009964548
Epoch [16]: Loss 0.010249966
Epoch [16]: Loss 0.010381383
Validation: Loss 0.009350598 Accuracy 1.0
Validation: Loss 0.009623845 Accuracy 1.0
Epoch [17]: Loss 0.009439203
Epoch [17]: Loss 0.00945879
Epoch [17]: Loss 0.009169875
Epoch [17]: Loss 0.009273212
Epoch [17]: Loss 0.009408832
Epoch [17]: Loss 0.0094184475
Epoch [17]: Loss 0.009200359
Validation: Loss 0.008535057 Accuracy 1.0
Validation: Loss 0.008785743 Accuracy 1.0
Epoch [18]: Loss 0.00830015
Epoch [18]: Loss 0.008380797
Epoch [18]: Loss 0.009379252
Epoch [18]: Loss 0.008420665
Epoch [18]: Loss 0.008459065
Epoch [18]: Loss 0.008304151
Epoch [18]: Loss 0.009140285
Validation: Loss 0.007842325 Accuracy 1.0
Validation: Loss 0.00807429 Accuracy 1.0
Epoch [19]: Loss 0.008597275
Epoch [19]: Loss 0.0074982047
Epoch [19]: Loss 0.007896465
Epoch [19]: Loss 0.0076710684
Epoch [19]: Loss 0.0075154766
Epoch [19]: Loss 0.008188119
Epoch [19]: Loss 0.0077530397
Validation: Loss 0.0072431965 Accuracy 1.0
Validation: Loss 0.0074592065 Accuracy 1.0
Epoch [20]: Loss 0.0075809183
Epoch [20]: Loss 0.0076649725
Epoch [20]: Loss 0.0074122176
Epoch [20]: Loss 0.0070295325
Epoch [20]: Loss 0.007155309
Epoch [20]: Loss 0.0070452923
Epoch [20]: Loss 0.0070338678
Validation: Loss 0.006722062 Accuracy 1.0
Validation: Loss 0.0069227363 Accuracy 1.0
Epoch [21]: Loss 0.0067964047
Epoch [21]: Loss 0.0070549003
Epoch [21]: Loss 0.006607674
Epoch [21]: Loss 0.0071153436
Epoch [21]: Loss 0.006575522
Epoch [21]: Loss 0.0066391476
Epoch [21]: Loss 0.0065604923
Validation: Loss 0.0062648207 Accuracy 1.0
Validation: Loss 0.0064535993 Accuracy 1.0
Epoch [22]: Loss 0.0063841315
Epoch [22]: Loss 0.0067780176
Epoch [22]: Loss 0.006238669
Epoch [22]: Loss 0.006591294
Epoch [22]: Loss 0.006263328
Epoch [22]: Loss 0.005746831
Epoch [22]: Loss 0.0064556412
Validation: Loss 0.005859765 Accuracy 1.0
Validation: Loss 0.0060364283 Accuracy 1.0
Epoch [23]: Loss 0.006187301
Epoch [23]: Loss 0.0060812575
Epoch [23]: Loss 0.0058444072
Epoch [23]: Loss 0.0061056996
Epoch [23]: Loss 0.00549619
Epoch [23]: Loss 0.0059944163
Epoch [23]: Loss 0.0055629457
Validation: Loss 0.005497816 Accuracy 1.0
Validation: Loss 0.0056649833 Accuracy 1.0
Epoch [24]: Loss 0.0057872543
Epoch [24]: Loss 0.0059626345
Epoch [24]: Loss 0.0052672126
Epoch [24]: Loss 0.005334369
Epoch [24]: Loss 0.005798745
Epoch [24]: Loss 0.0053602313
Epoch [24]: Loss 0.0053820796
Validation: Loss 0.005174194 Accuracy 1.0
Validation: Loss 0.005331272 Accuracy 1.0
Epoch [25]: Loss 0.005278251
Epoch [25]: Loss 0.005608474
Epoch [25]: Loss 0.005446313
Epoch [25]: Loss 0.005289626
Epoch [25]: Loss 0.0052443305
Epoch [25]: Loss 0.0048154267
Epoch [25]: Loss 0.0046561873
Validation: Loss 0.004881327 Accuracy 1.0
Validation: Loss 0.0050313217 Accuracy 1.0

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
  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.130 GiB / 4.750 GiB available)
┌ Warning: LuxAMDGPU is loaded but the AMDGPU is not functional.
└ @ LuxAMDGPU ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/LuxAMDGPU/sGa0S/src/LuxAMDGPU.jl:19

```


---


*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

