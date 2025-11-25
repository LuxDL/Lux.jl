


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
    anticlockwise_spirals = [reshape(d[1][:, (sequence_length + 1):end], :,
        sequence_length, 1) for d in data[((dataset_size ÷ 2) + 1):end]]
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


To understand more about container layers, please look at [Container Layer](http://lux.csail.mit.edu/stable/manual/interface/#container-layer).


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
    return SpiralClassifier(LSTMCell(in_dims => hidden_dims),
        Dense(hidden_dims => out_dims, sigmoid))
end
```


```
Main.var"##225".SpiralClassifier
```


We can use default Lux blocks – `Recurrence(LSTMCell(in_dims => hidden_dims)` – instead of defining the following. But let's still do it for the sake of it.


Now we need to define the behavior of the Classifier when it is invoked.


```julia
function (s::SpiralClassifier)(x::AbstractArray{T, 3}, ps::NamedTuple,
        st::NamedTuple) where {T}
    # First we will have to run the sequence through the LSTM Cell
    # The first call to LSTM Cell will create the initial hidden state
    # See that the parameters and states are automatically populated into a field called
    # `lstm_cell` We use `eachslice` to get the elements in the sequence without copying,
    # and `Iterators.peel` to split out the first element for LSTM initialization.
    x_init, x_rest = Iterators.peel(eachslice(x; dims=2))
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

matches(y_pred, y_true) = sum((y_pred .> 0.5) .== y_true)
accuracy(y_pred, y_true) = matches(y_pred, y_true) / length(y_pred)
```


```
accuracy (generic function with 1 method)
```


Finally lets create an optimiser given the model parameters.


```julia
function create_optimiser(ps)
    opt = Optimisers.ADAM(0.01f0)
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
((lstm_cell = (weight_i = Float32[-0.8629139 -0.4196577; -0.2536159 -0.75585; 0.81421643 0.88533586; -0.0623762 0.07550464; -0.14470567 -0.62319773; 0.28592616 0.55141324; -0.8371497 0.07844722; 0.068350844 0.053690854; -0.040071644 0.07323416; -0.6641252 0.451696; 1.1784786 -0.017274735; -0.01829934 0.0885153; -0.11234444 0.24584728; -0.8522731 0.3187137; -0.12762125 -0.27130276; 1.2113008 0.1047074; 0.68168217 -0.6265603; 0.12882888 -0.343052; 0.46233615 -0.32326502; -0.5504713 0.5191638; 0.58195555 -0.8371756; 0.09713171 0.292289; 0.8713551 -0.63823104; 0.95642537 0.62687886; -1.2623378 -0.08810054; 0.47621337 0.64224195; 1.05528 0.6740735; -0.44963083 -0.1443426; 0.74564904 -0.69429976; -0.34385473 0.71950465; -0.22072004 0.61013865; 0.6302593 -0.28034934], weight_h = Float32[-0.5189122 -0.05688713 0.30404913 -0.26291507 0.29490444 0.011861273 -0.7150106 0.5470436; -0.65604323 0.20809016 -0.06102144 0.62761396 0.37924212 0.3087763 0.13143691 -0.0440198; 0.012284191 0.063882984 -0.033234134 0.56727886 -0.7378445 0.31556624 -0.6423968 -0.14507368; 0.035878453 -0.264465 0.88879216 -0.09005087 0.91113067 0.10249612 0.27938902 0.90770054; -0.3742098 0.46821955 0.7815788 0.32635328 0.40164673 0.6334855 -0.33589545 -0.0779683; -0.060489032 -0.33459723 0.33075264 -0.093676664 -0.28850108 -0.21782276 -0.49007073 0.03494646; -0.7933202 0.3751013 0.6863645 0.52291554 -0.81219304 0.63772994 0.020895446 -0.61725354; 0.6838676 0.30973998 1.0914423 -1.302183 0.8000643 0.16677783 -0.7223795 1.1377636; -0.17501068 0.4722302 0.42522192 -0.49545535 0.4574798 0.68045974 0.17150939 0.6335295; -0.41973677 -0.18564829 -0.38406643 -0.006971132 0.23909831 -0.002406527 -0.753244 0.74890167; -0.15202226 0.5535727 -0.034637503 0.5275261 -0.5133314 0.29555663 -0.27003726 -0.36338302; 0.053976063 0.8853655 0.28867432 -0.10607433 0.8278068 0.6538051 0.38723445 0.49877757; 0.6473319 0.4922825 0.3731544 -0.06903875 0.921778 0.16166446 -0.70538586 0.44197422; -0.09699746 0.44421393 0.08876722 0.096137024 0.60563946 -0.016523933 -0.14278002 -0.40007985; -0.82769233 0.32836702 0.081707455 -0.393434 -0.26482987 0.80345124 -0.37294462 0.42466226; -0.6369352 0.78806055 0.41342044 1.0533568 -0.09033825 0.8145932 -0.09761213 0.75946707; 0.14816989 -0.64177245 -0.052784145 -0.4438386 -0.35262063 -0.22000732 -0.16526514 0.17024907; -0.44196552 0.23092009 0.19575879 0.7181952 0.3923992 -0.35614777 -0.36069193 0.6626692; -0.55672276 0.70870656 0.093885876 -0.42784607 -0.3139001 0.74300367 0.04795101 0.6061253; -0.7779215 0.3394865 -0.16824968 0.5083617 -0.6995191 -0.13592488 -0.20023586 -0.19781956; -0.26749584 -0.6640775 0.39969146 -0.6436915 -0.16483715 0.26521528 0.2491417 -0.12508479; -0.7258991 -0.42887136 0.5360144 -0.48000246 -0.13453344 0.030599948 -0.44411165 0.21885797; 0.48526442 -0.19161907 -0.66022456 0.25718108 0.34148836 0.072735325 -0.52320147 -0.41823658; -0.42127863 0.35831377 0.22719797 0.37940064 -0.2793088 -0.18539216 -0.7532595 0.07834724; -0.66450804 0.5479232 0.26852274 -0.4398687 0.59334356 0.4320564 -0.9598233 0.8681743; -0.6032221 0.26773995 0.07233512 0.42822474 -0.26683423 0.5617497 -0.41486466 -0.28841618; -0.819564 -0.27391848 0.4614911 -0.2740082 0.25138247 0.011596974 -0.89477855 0.75716436; 0.2472241 0.27503955 1.0199367 -0.8973961 0.24904066 0.18185265 -1.1397282 0.49609968; -0.385685 0.27643985 0.7698755 -0.40833536 0.44688052 0.8940588 -1.1969987 0.7190273; 0.4564864 0.38265347 -0.6952036 0.64404684 -1.2560395 -0.38668063 -0.1507112 -0.06572576; -0.4349146 0.5171811 -0.2607722 0.7459363 -0.81807464 -0.13282004 0.08745216 -0.0779775; -0.5615953 0.7345092 0.6704763 -0.50442165 0.66447765 0.04935179 -0.4863874 0.6401504], bias = Float32[0.28793597; 0.26808345; 0.14582318; 0.32513213; 0.342989; 0.12164306; 0.0119883595; 0.9676595; 0.37753224; 0.15714988; 0.011338698; 0.35670447; 0.43247685; 0.06888572; -0.013767286; 0.3181378; 0.84286135; 1.1313248; 1.1593057; 0.76705027; 0.69000155; 1.2353724; 0.6399854; 0.90624386; 0.472126; 0.060084112; 0.307313; 0.62572086; 0.6358444; 0.0077448874; -0.117289424; 0.876123;;]), classifier = (weight = Float32[-1.4280366 0.76387614 1.2379253 1.2582616 -0.93207204 0.1151196 -0.26813972 1.2173434], bias = Float32[-0.6929531;;])), (lstm_cell = (rng = Random.Xoshiro(0x2026f555c226bf09, 0x8a6bb764b93cadda, 0x5ba3c10439600514, 0x446f763658f71987, 0x22a21880af5dc689),), classifier = NamedTuple()))
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


---


*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

