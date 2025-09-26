


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
function (s::SpiralClassifier)(x::AbstractArray{T, 3},
    ps::NamedTuple,
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
            (loss, y_pred, st), back = pullback(p -> compute_loss(x, y, model, p, st), ps)
            gs = back((one(loss), nothing, nothing))[1]
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
((lstm_cell = (weight_i = Float32[-0.8621917 -0.42038837; -0.25192982 -0.7572166; 0.7976483 0.87956375; -0.05165504 0.07698642; -0.14871594 -0.61985123; 0.29320064 0.5539806; -0.8302107 0.060381; 0.080584995 0.062160745; -0.038774215 0.089452125; -0.65823275 0.45045033; 1.1244819 -0.03172796; -0.008779214 0.088383935; -0.09999141 0.24514697; -0.8451995 0.33073518; -0.13284227 -0.2816304; 1.2285876 0.0918826; 0.6748522 -0.6312087; 0.13384132 -0.3496426; 0.47796568 -0.32991806; -0.5641144 0.5152274; 0.5849884 -0.83493644; 0.109559625 0.2993496; 0.86355454 -0.6398492; 0.94000024 0.6105737; -1.2273667 -0.09762106; 0.4939085 0.6435425; 1.0574492 0.680253; -0.464749 -0.17523645; 0.72662526 -0.7067004; -0.3302923 0.75044197; -0.21786581 0.6139128; 0.5983437 -0.32036108], weight_h = Float32[-0.51444215 -0.052080564 0.2720881 -0.25705308 0.3033216 0.0032504634 -0.69410425 0.5553928; -0.6417825 0.24484698 -0.06909503 0.6420119 0.39192376 0.31350443 0.13165747 -0.04704204; 0.023109192 0.057685424 -0.03783412 0.55345994 -0.7479571 0.325942 -0.6276448 -0.17787772; 0.035547417 -0.28018805 0.8841569 -0.09262122 0.91116756 0.066858925 0.27468938 0.90688854; -0.367504 0.47375712 0.74294674 0.30326524 0.40236807 0.6385031 -0.34387472 -0.07950349; -0.05865665 -0.34033194 0.33341014 -0.09897807 -0.28983998 -0.22313844 -0.49925175 0.051198706; -0.7772122 0.3676499 0.7062585 0.48486757 -0.7998102 0.6363063 0.039359104 -0.62741476; 0.68261063 0.3034421 0.99977964 -1.3228431 0.83715206 0.14736933 -0.69305503 1.1601193; -0.1409507 0.4741748 0.39298263 -0.5070918 0.4691759 0.6721045 0.20258439 0.64792866; -0.41547045 -0.19010228 -0.3794329 -0.01668561 0.2495109 0.002474777 -0.7456068 0.7595518; -0.1422577 0.5486426 -0.08472607 0.52798414 -0.5337521 0.29018387 -0.26278317 -0.3899452; 0.05521976 0.8791162 0.2862201 -0.105660394 0.8280769 0.5890401 0.39058095 0.50001186; 0.6507861 0.49364474 0.3720466 -0.07295907 0.9234742 0.18293343 -0.6750142 0.4437327; -0.10601119 0.44191864 0.08109501 0.07640825 0.60838854 -0.013756296 -0.12836692 -0.4313815; -0.843652 0.31692246 0.09962912 -0.41087306 -0.2534173 0.8055427 -0.35906622 0.41021279; -0.6562092 0.79220146 0.42007047 1.0475487 -0.09027772 0.8196584 -0.11340295 0.7667307; 0.15393576 -0.63410366 -0.050198313 -0.43353423 -0.35080504 -0.20532344 -0.15953398 0.18248722; -0.43498394 0.22531357 0.20020676 0.7112834 0.39755058 -0.35115585 -0.36353374 0.6908401; -0.5499527 0.707 0.09483351 -0.43620476 -0.33666298 0.7369176 0.05428296 0.5974111; -0.7722187 0.3411446 -0.16620132 0.49871546 -0.69434035 -0.15838963 -0.19894409 -0.19594157; -0.272523 -0.6693816 0.41236752 -0.6455297 -0.17270757 0.26647788 0.24570519 -0.12074696; -0.725112 -0.42518923 0.5255903 -0.47501785 -0.17213039 0.026157185 -0.43932313 0.22031882; 0.48394665 -0.18692347 -0.66864246 0.29635534 0.34327498 0.06946309 -0.5237582 -0.41063914; -0.407949 0.35211104 0.2042309 0.36051798 -0.3202218 -0.20015126 -0.73932236 -0.054656662; -0.6371878 0.5559082 0.25225472 -0.46309313 0.6076663 0.43650123 -0.9182645 0.8740231; -0.6056121 0.2787688 0.05676565 0.43002313 -0.26909867 0.5662737 -0.4139492 -0.33909035; -0.8212502 -0.27391595 0.45969823 -0.27024198 0.21622232 0.01852199 -0.8823942 0.742675; 0.3097836 0.26138797 1.0090545 -0.905068 0.24254963 0.1677703 -1.081484 0.46641102; -0.34388822 0.28031173 0.76859856 -0.4475207 0.4556283 0.9118428 -1.1803097 0.6991986; 0.42424187 0.4069453 -0.6917944 0.58762395 -1.1115111 -0.36753306 -0.16325426 -0.05984559; -0.43314862 0.51591575 -0.24104853 0.6860696 -0.8202701 -0.14483435 0.08108103 -0.09687297; -0.54565895 0.7675619 0.70328957 -0.5524565 0.6222204 0.05728875 -0.4820327 0.6616742], bias = Float32[0.295456; 0.29876825; 0.1343874; 0.32046142; 0.34513348; 0.11471745; 0.012313617; 0.98749596; 0.3863972; 0.15178625; 0.0035769728; 0.3558427; 0.43393484; 0.06736191; -0.014284401; 0.31570742; 0.8495216; 1.1262524; 1.152703; 0.77123827; 0.6879085; 1.2394948; 0.6486227; 0.9243773; 0.47972596; 0.043080002; 0.29976058; 0.6108152; 0.6355568; 0.031105174; -0.12007648; 0.87487763;;]), classifier = (weight = Float32[-1.4426658 0.75237757 1.2464963 1.2675542 -0.94357157 0.11813501 -0.26595384 1.228135], bias = Float32[-0.5505798;;])), (lstm_cell = (rng = Random.Xoshiro(0x2026f555c226bf09, 0x8a6bb764b93cadda, 0x5ba3c10439600514, 0x446f763658f71987),), classifier = NamedTuple()))
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

