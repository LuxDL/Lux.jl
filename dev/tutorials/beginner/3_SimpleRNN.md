---
url: /dev/tutorials/beginner/3_SimpleRNN.md
---

# Training a Simple LSTM {#Training-a-Simple-LSTM}

In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:
2\. Create custom Lux models.

3. Become familiar with the Lux recurrent neural network API.

4. Training using Optimisers.jl and Zygote.jl.

## Package Imports {#Package-Imports}

Note: If you wish to use `AutoZygote()` for automatic differentiation, add Zygote to your project dependencies and include `using Zygote`.

```julia
using ADTypes, Lux, JLD2, MLUtils, Optimisers, Printf, Reactant, Random
```

## Dataset {#Dataset}

We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a `MLUtils.DataLoader`. Our dataloader will give us sequences of size 2 × seq\_len × batch\_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.

```julia
function create_dataset(; dataset_size=1000, sequence_length=50)
    # Create the spirals
    data = [MLUtils.Datasets.make_spiral(sequence_length) for _ in 1:dataset_size]
    # Get the labels
    labels = vcat(repeat([0.0f0], dataset_size ÷ 2), repeat([1.0f0], dataset_size ÷ 2))
    clockwise_spirals = [
        reshape(d[1][:, 1:sequence_length], :, sequence_length, 1) for
        d in data[1:(dataset_size ÷ 2)]
    ]
    anticlockwise_spirals = [
        reshape(d[1][:, (sequence_length + 1):end], :, sequence_length, 1) for
        d in data[((dataset_size ÷ 2) + 1):end]
    ]
    x_data = Float32.(cat(clockwise_spirals..., anticlockwise_spirals...; dims=3))
    return x_data, labels
end

function get_dataloaders(; dataset_size=1000, sequence_length=50)
    x_data, labels = create_dataset(; dataset_size, sequence_length)
    # Split the dataset
    (x_train, y_train), (x_val, y_val) = splitobs((x_data, labels); at=0.8, shuffle=true)
    # Create DataLoaders
    return (
        # Use DataLoader to automatically minibatch and shuffle the data
        DataLoader(
            collect.((x_train, y_train)); batchsize=128, shuffle=true, partial=false
        ),
        # Don't shuffle the validation data
        DataLoader(collect.((x_val, y_val)); batchsize=128, shuffle=false, partial=false),
    )
end
```

## Creating a Classifier {#Creating-a-Classifier}

We will be extending the `Lux.AbstractLuxContainerLayer` type for our custom model since it will contain a LSTM block and a classifier head.

We pass the field names `lstm_cell` and `classifier` to the type to ensure that the parameters and states are automatically populated and we don't have to define `Lux.initialparameters` and `Lux.initialstates`.

To understand more about container layers, please look at [Container Layer](/manual/interface#Container-Layer).

```julia
struct SpiralClassifier{L,C} <: AbstractLuxContainerLayer{(:lstm_cell, :classifier)}
    lstm_cell::L
    classifier::C
end
```

We won't define the model from scratch but rather use the [`Lux.LSTMCell`](/api/Lux/layers#Lux.LSTMCell) and [`Lux.Dense`](/api/Lux/layers#Lux.Dense).

```julia
function SpiralClassifier(in_dims, hidden_dims, out_dims)
    return SpiralClassifier(
        LSTMCell(in_dims => hidden_dims), Dense(hidden_dims => out_dims, sigmoid)
    )
end
```

We can use default Lux blocks – `Recurrence(LSTMCell(in_dims => hidden_dims)` – instead of defining the following. But let's still do it for the sake of it.

Now we need to define the behavior of the Classifier when it is invoked.

```julia
function (s::SpiralClassifier)(
    x::AbstractArray{T,3}, ps::NamedTuple, st::NamedTuple
) where {T}
    # First we will have to run the sequence through the LSTM Cell
    # The first call to LSTM Cell will create the initial hidden state
    # See that the parameters and states are automatically populated into a field called
    # `lstm_cell` We use `eachslice` to get the elements in the sequence without copying,
    # and `Iterators.peel` to split out the first element for LSTM initialization.
    x_init, x_rest = Iterators.peel(LuxOps.eachslice(x, Val(2)))
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

## Using the `@compact` API {#Using-the-@compact-API}

We can also define the model using the [`Lux.@compact`](/api/Lux/utilities#Lux.@compact) API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers

```julia
function SpiralClassifierCompact(in_dims, hidden_dims, out_dims)
    return @compact(;
        lstm_cell=LSTMCell(in_dims => hidden_dims),
        classifier=Dense(hidden_dims => out_dims, sigmoid)
    ) do x::AbstractArray{T,3} where {T}
        x_init, x_rest = Iterators.peel(LuxOps.eachslice(x, Val(2)))
        y, carry = lstm_cell(x_init)
        for x in x_rest
            y, carry = lstm_cell((x, carry))
        end
        @return vec(classifier(y))
    end
end
```

## Defining Accuracy, Loss and Optimiser {#Defining-Accuracy,-Loss-and-Optimiser}

Now let's define the binary cross-entropy loss. Typically it is recommended to use `logitbinarycrossentropy` since it is more numerically stable, but for the sake of simplicity we will use `binarycrossentropy`.

```julia
const lossfn = BinaryCrossEntropyLoss()

function compute_loss(model, ps, st, (x, y))
    ŷ, st_ = model(x, ps, st)
    loss = lossfn(ŷ, y)
    return loss, st_, (; y_pred=ŷ)
end

matches(y_pred, y_true) = sum((y_pred .> 0.5f0) .== y_true)
accuracy(y_pred, y_true) = matches(y_pred, y_true) / length(y_pred)
```

## Training the Model {#Training-the-Model}

```julia
function main(model_type)
    dev = reactant_device()
    cdev = cpu_device()

    # Get the dataloaders
    train_loader, val_loader = get_dataloaders() |> dev

    # Create the model
    model = model_type(2, 8, 1)
    ps, st = Lux.setup(Random.default_rng(), model) |> dev

    train_state = Training.TrainState(model, ps, st, Adam(0.01f0))
    model_compiled = if dev isa ReactantDevice
        @compile model(first(train_loader)[1], ps, Lux.testmode(st))
    else
        model
    end
    ad = dev isa ReactantDevice ? AutoReactant() : AutoZygote()

    for epoch in 1:25
        # Train the model
        total_loss = 0.0f0
        total_samples = 0
        for (x, y) in train_loader
            (_, loss, _, train_state) = Training.single_train_step!(
                ad, lossfn, (x, y), train_state
            )
            total_loss += loss * length(y)
            total_samples += length(y)
        end
        @printf("Epoch [%3d]: Loss %4.5f\n", epoch, total_loss / total_samples)

        # Validate the model
        total_acc = 0.0f0
        total_loss = 0.0f0
        total_samples = 0

        st_ = Lux.testmode(train_state.states)
        for (x, y) in val_loader
            ŷ, st_ = model_compiled(x, train_state.parameters, st_)
            ŷ, y = cdev(ŷ), cdev(y)
            total_acc += accuracy(ŷ, y) * length(y)
            total_loss += lossfn(ŷ, y) * length(y)
            total_samples += length(y)
        end

        @printf(
            "Validation:\tLoss %4.5f\tAccuracy %4.5f\n",
            total_loss / total_samples,
            total_acc / total_samples
        )
    end

    return (train_state.parameters, train_state.states) |> cdev
end

ps_trained, st_trained = main(SpiralClassifier)
```

```
┌ Warning: `replicate` doesn't work for `TaskLocalRNG`. Returning the same `TaskLocalRNG`.
└ @ LuxCore ~/work/Lux.jl/Lux.jl/lib/LuxCore/src/LuxCore.jl:18
Epoch [  1]: Loss 0.69692
Validation:	Loss 0.58114	Accuracy 0.60938
Epoch [  2]: Loss 0.52475
Validation:	Loss 0.44499	Accuracy 1.00000
Epoch [  3]: Loss 0.40445
Validation:	Loss 0.34096	Accuracy 1.00000
Epoch [  4]: Loss 0.30953
Validation:	Loss 0.25821	Accuracy 1.00000
Epoch [  5]: Loss 0.23474
Validation:	Loss 0.19400	Accuracy 1.00000
Epoch [  6]: Loss 0.17551
Validation:	Loss 0.14412	Accuracy 1.00000
Epoch [  7]: Loss 0.12758
Validation:	Loss 0.10044	Accuracy 1.00000
Epoch [  8]: Loss 0.08603
Validation:	Loss 0.06668	Accuracy 1.00000
Epoch [  9]: Loss 0.05810
Validation:	Loss 0.04673	Accuracy 1.00000
Epoch [ 10]: Loss 0.04158
Validation:	Loss 0.03535	Accuracy 1.00000
Epoch [ 11]: Loss 0.03228
Validation:	Loss 0.02831	Accuracy 1.00000
Epoch [ 12]: Loss 0.02618
Validation:	Loss 0.02339	Accuracy 1.00000
Epoch [ 13]: Loss 0.02181
Validation:	Loss 0.01984	Accuracy 1.00000
Epoch [ 14]: Loss 0.01868
Validation:	Loss 0.01720	Accuracy 1.00000
Epoch [ 15]: Loss 0.01635
Validation:	Loss 0.01517	Accuracy 1.00000
Epoch [ 16]: Loss 0.01450
Validation:	Loss 0.01356	Accuracy 1.00000
Epoch [ 17]: Loss 0.01300
Validation:	Loss 0.01225	Accuracy 1.00000
Epoch [ 18]: Loss 0.01177
Validation:	Loss 0.01116	Accuracy 1.00000
Epoch [ 19]: Loss 0.01078
Validation:	Loss 0.01023	Accuracy 1.00000
Epoch [ 20]: Loss 0.00987
Validation:	Loss 0.00944	Accuracy 1.00000
Epoch [ 21]: Loss 0.00915
Validation:	Loss 0.00874	Accuracy 1.00000
Epoch [ 22]: Loss 0.00849
Validation:	Loss 0.00813	Accuracy 1.00000
Epoch [ 23]: Loss 0.00791
Validation:	Loss 0.00759	Accuracy 1.00000
Epoch [ 24]: Loss 0.00739
Validation:	Loss 0.00710	Accuracy 1.00000
Epoch [ 25]: Loss 0.00693
Validation:	Loss 0.00667	Accuracy 1.00000

```

We can also train the compact model with the exact same code!

```julia
ps_trained2, st_trained2 = main(SpiralClassifierCompact)
```

```
┌ Warning: `replicate` doesn't work for `TaskLocalRNG`. Returning the same `TaskLocalRNG`.
└ @ LuxCore ~/work/Lux.jl/Lux.jl/lib/LuxCore/src/LuxCore.jl:18
Epoch [  1]: Loss 0.43529
Validation:	Loss 0.36586	Accuracy 1.00000
Epoch [  2]: Loss 0.32680
Validation:	Loss 0.28040	Accuracy 1.00000
Epoch [  3]: Loss 0.24799
Validation:	Loss 0.21099	Accuracy 1.00000
Epoch [  4]: Loss 0.18331
Validation:	Loss 0.15156	Accuracy 1.00000
Epoch [  5]: Loss 0.13006
Validation:	Loss 0.10662	Accuracy 1.00000
Epoch [  6]: Loss 0.09243
Validation:	Loss 0.07589	Accuracy 1.00000
Epoch [  7]: Loss 0.06607
Validation:	Loss 0.05482	Accuracy 1.00000
Epoch [  8]: Loss 0.04810
Validation:	Loss 0.04016	Accuracy 1.00000
Epoch [  9]: Loss 0.03571
Validation:	Loss 0.03030	Accuracy 1.00000
Epoch [ 10]: Loss 0.02729
Validation:	Loss 0.02423	Accuracy 1.00000
Epoch [ 11]: Loss 0.02225
Validation:	Loss 0.02010	Accuracy 1.00000
Epoch [ 12]: Loss 0.01851
Validation:	Loss 0.01664	Accuracy 1.00000
Epoch [ 13]: Loss 0.01519
Validation:	Loss 0.01367	Accuracy 1.00000
Epoch [ 14]: Loss 0.01278
Validation:	Loss 0.01166	Accuracy 1.00000
Epoch [ 15]: Loss 0.01101
Validation:	Loss 0.01024	Accuracy 1.00000
Epoch [ 16]: Loss 0.00976
Validation:	Loss 0.00913	Accuracy 1.00000
Epoch [ 17]: Loss 0.00871
Validation:	Loss 0.00821	Accuracy 1.00000
Epoch [ 18]: Loss 0.00786
Validation:	Loss 0.00743	Accuracy 1.00000
Epoch [ 19]: Loss 0.00714
Validation:	Loss 0.00677	Accuracy 1.00000
Epoch [ 20]: Loss 0.00651
Validation:	Loss 0.00619	Accuracy 1.00000
Epoch [ 21]: Loss 0.00596
Validation:	Loss 0.00567	Accuracy 1.00000
Epoch [ 22]: Loss 0.00547
Validation:	Loss 0.00521	Accuracy 1.00000
Epoch [ 23]: Loss 0.00503
Validation:	Loss 0.00479	Accuracy 1.00000
Epoch [ 24]: Loss 0.00462
Validation:	Loss 0.00441	Accuracy 1.00000
Epoch [ 25]: Loss 0.00427
Validation:	Loss 0.00408	Accuracy 1.00000

```

## Saving the Model {#Saving-the-Model}

We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don't save the model struct and only save the parameters and states.

```julia
@save "trained_model.jld2" ps_trained st_trained
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
