


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
Main.var"##292".SpiralClassifier
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
((lstm_cell = (weight_i = Float32[-0.86135656 -0.42335424; -0.2525119 -0.75629354; 0.79978925 0.8833773; -0.05418714 0.07787614; -0.14919703 -0.62482053; 0.29151326 0.55013233; -0.8277675 0.061439253; 0.084517084 0.06017387; -0.041264817 0.09163789; -0.66007555 0.45158365; 1.1348319 -0.027801292; -0.011530283 0.089988135; -0.10216045 0.24818121; -0.8483534 0.32604465; -0.13086198 -0.28246257; 1.2153202 0.08776247; 0.67674845 -0.6300841; 0.1316891 -0.34767944; 0.48271382 -0.3293832; -0.5615602 0.51683146; 0.5844544 -0.83695006; 0.105046034 0.2956856; 0.86389846 -0.64043736; 0.9412701 0.6054893; -1.2820674 -0.09797412; 0.4868299 0.64760566; 1.0560008 0.6802865; -0.47183505 -0.1786497; 0.7188753 -0.71203583; -0.33077744 0.75680155; -0.21348397 0.61600834; 0.5992792 -0.32726598], weight_h = Float32[-0.51286936 -0.05135065 0.2775495 -0.2512987 0.30437076 0.00460656 -0.68829006 0.5556629; -0.6493038 0.23568702 -0.06723616 0.6342212 0.3891629 0.31225654 0.13117842 -0.05820769; 0.018973714 0.061439313 -0.024947086 0.5530931 -0.7478726 0.32751268 -0.6286055 -0.18760969; 0.037215114 -0.27776212 0.88102096 -0.0918686 0.909383 0.06978588 0.27956766 0.90551263; -0.36364692 0.47387892 0.7642875 0.3021619 0.4022823 0.64138865 -0.34504738 -0.08054078; -0.05568735 -0.33998486 0.3332937 -0.09913609 -0.2856072 -0.22163683 -0.49122018 0.056484096; -0.77952933 0.36724553 0.7049048 0.48537862 -0.79727215 0.63870573 0.0348326 -0.63141847; 0.6795884 0.29579628 0.9915114 -1.3294098 0.83991116 0.15398799 -0.69858825 1.1548634; -0.13665633 0.47336337 0.3948419 -0.5114039 0.46673417 0.6733306 0.20354785 0.64376205; -0.41664144 -0.18859111 -0.37998322 -0.014582658 0.24763773 0.0016115957 -0.75531065 0.72252995; -0.14411598 0.54853266 -0.07014593 0.52616507 -0.5312125 0.2918717 -0.26507592 -0.39538097; 0.056453172 0.87591153 0.28276622 -0.102743454 0.8260601 0.5907367 0.3954548 0.49751773; 0.65109646 0.4900993 0.37116164 -0.07119748 0.92034405 0.19929817 -0.6145725 0.44131425; -0.10618301 0.44546488 0.083223924 0.079874665 0.6132119 -0.008523853 -0.13123254 -0.4102445; -0.83666444 0.31982195 0.095628984 -0.4112721 -0.25151554 0.8120642 -0.36258593 0.405801; -0.65797734 0.7863039 0.41032177 1.0418485 -0.09027146 0.8172089 -0.111068726 0.75465566; 0.15099415 -0.6354557 -0.042538974 -0.4343569 -0.3521624 -0.2074265 -0.16129024 0.18539463; -0.43705094 0.22729741 0.19836235 0.713122 0.40096822 -0.3519559 -0.3630698 0.6780172; -0.5503618 0.70478714 0.09387195 -0.43542346 -0.33838695 0.7374535 0.053538665 0.59530634; -0.77399147 0.33916345 -0.16796651 0.4994262 -0.69566405 -0.16121045 -0.20115165 -0.19677208; -0.27347347 -0.6672705 0.4057025 -0.6408396 -0.17844847 0.26609096 0.2503055 -0.114392705; -0.71931356 -0.43040138 0.5267777 -0.4844971 -0.14527668 0.028979953 -0.43926966 0.2228571; 0.4854348 -0.19858706 -0.6731762 0.30729657 0.34159002 0.066019535 -0.52388746 -0.40918356; -0.40806755 0.3523637 0.20986632 0.35740197 -0.3448113 -0.20085132 -0.7387854 -0.07211999; -0.64299613 0.55859053 0.2619767 -0.46647426 0.60277665 0.43950862 -0.92688906 0.86863583; -0.6090008 0.27937764 0.06215838 0.43032625 -0.2713065 0.5699229 -0.41845223 -0.35561952; -0.82367593 -0.26636147 0.4666233 -0.27233866 0.22021079 0.019561071 -0.88753873 0.7526717; 0.3096034 0.26504183 1.0136358 -0.90807635 0.23939434 0.18428569 -1.0829446 0.46002394; -0.3313644 0.2829505 0.7733299 -0.43133995 0.4509841 0.9350744 -1.1791824 0.68817675; 0.41102976 0.41768584 -0.68936557 0.5902695 -1.1227639 -0.3557743 -0.16911201 -0.057151902; -0.43482608 0.5173169 -0.24048403 0.6818809 -0.820355 -0.13794112 0.07831002 -0.10034156; -0.5451036 0.7692721 0.708982 -0.55877906 0.5999279 0.060453285 -0.48320717 0.65435755], bias = Float32[0.29691505; 0.290489; 0.1362606; 0.32009712; 0.34493786; 0.116268165; 0.01325658; 0.9864648; 0.38658792; 0.15337116; 0.0034737454; 0.35370502; 0.4306089; 0.07107551; -0.013885329; 0.31438965; 0.84832335; 1.1283482; 1.153218; 0.76961; 0.68895537; 1.2380905; 0.64889765; 0.9309451; 0.47878903; 0.034986313; 0.30882442; 0.61053056; 0.6301711; 0.038949583; -0.12052576; 0.8719889;;]), classifier = (weight = Float32[-1.4439584 0.7814694 1.2526563 1.2652947 -0.9392555 0.1206586 -0.2681623 1.21811], bias = Float32[-0.5556541;;])), (lstm_cell = (rng = Random.Xoshiro(0x2026f555c226bf09, 0x8a6bb764b93cadda, 0x5ba3c10439600514, 0x446f763658f71987),), classifier = NamedTuple()))
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

