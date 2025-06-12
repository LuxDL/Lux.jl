


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
((lstm_cell = (weight_i = Float32[-0.86194944 -0.41941056; -0.2522536 -0.75545156; 0.8046613 0.88755596; -0.05660284 0.07870288; -0.14501388 -0.62312025; 0.29035452 0.5506007; -0.8312554 0.06700357; 0.084451 0.05263913; -0.042531557 0.08483349; -0.66053253 0.45123535; 1.1327479 -0.022961615; -0.01461209 0.08916909; -0.10148164 0.24800417; -0.8489461 0.3256773; -0.13058604 -0.27624094; 1.212919 0.08795164; 0.67834884 -0.6306514; 0.13102017 -0.34739628; 0.46428183 -0.3278071; -0.5566078 0.51528955; 0.584727 -0.83691645; 0.10001048 0.29366744; 0.86667883 -0.6398646; 0.94742393 0.6109794; -1.2711544 -0.103596404; 0.4865266 0.64476293; 1.0550731 0.6812794; -0.46049243 -0.1731563; 0.7238055 -0.70950663; -0.33521518 0.75187504; -0.21297894 0.6131987; 0.60165167 -0.32419428], weight_h = Float32[-0.51046455 -0.051870476 0.2824686 -0.25531882 0.3000278 0.009336694 -0.6862247 0.55116785; -0.6533524 0.23864368 -0.06466762 0.63351434 0.3880319 0.31232205 0.12933482 -0.05377329; 0.014031856 0.06267511 -0.042487588 0.55736256 -0.742838 0.32955572 -0.6341211 -0.18695986; 0.038331807 -0.27612036 0.8801711 -0.09698937 0.90917295 0.07690376 0.28012303 0.90512794; -0.3708926 0.4717805 0.7659082 0.3007279 0.4001938 0.6407235 -0.34720245 -0.08242243; -0.05701819 -0.3389476 0.33276057 -0.09805635 -0.28914043 -0.22053498 -0.4911356 0.04876995; -0.7849146 0.36981663 0.704337 0.49398986 -0.8028504 0.64124286 0.030770777 -0.63146615; 0.685331 0.30147493 1.0145698 -1.3270077 0.8309379 0.16071887 -0.7080812 1.1497253; -0.14243224 0.4747771 0.40416637 -0.5139905 0.46192122 0.67818093 0.19315603 0.63827723; -0.41688585 -0.18835756 -0.38028377 -0.01375926 0.24608243 0.0009582123 -0.75960463 0.7396329; -0.14725854 0.54936403 -0.06100933 0.5268104 -0.52437186 0.29320973 -0.26830447 -0.39138597; 0.05801682 0.8762195 0.28510243 -0.10948846 0.8278222 0.6064472 0.3945023 0.4991597; 0.6505157 0.49263892 0.3713625 -0.07312084 0.92147976 0.18916301 -0.61142623 0.44175762; -0.10560865 0.44520035 0.08629538 0.08349524 0.61057526 -0.009795159 -0.13462523 -0.40869123; -0.8328958 0.32396004 0.09061161 -0.40530494 -0.25547972 0.7996562 -0.3659486 0.40459168; -0.6650481 0.7903345 0.41232476 1.0860823 -0.0752047 0.8138168 -0.106666885 0.7727952; 0.1493441 -0.6369458 -0.04438855 -0.43581772 -0.35277426 -0.21573518 -0.16186796 0.17930062; -0.4373337 0.2276905 0.19828542 0.7135297 0.4013356 -0.3533775 -0.36379835 0.67497194; -0.55190504 0.70594215 0.09427455 -0.43413308 -0.3413539 0.7386469 0.052188285 0.5974973; -0.7734654 0.34049678 -0.1659193 0.49803984 -0.694188 -0.14706391 -0.20076525 -0.19405496; -0.2720589 -0.6685225 0.40482548 -0.64581376 -0.17043798 0.2635376 0.24912828 -0.119100004; -0.721109 -0.43099698 0.5306922 -0.48074988 -0.14826633 0.028848877 -0.43978015 0.21809737; 0.48502216 -0.19492984 -0.67064273 0.29216638 0.34008443 0.06822674 -0.5230616 -0.40754995; -0.41434756 0.35410473 0.22822657 0.36481926 -0.35608903 -0.1966037 -0.741412 0.016474087; -0.64702326 0.55824536 0.26381582 -0.46870813 0.5989327 0.44168606 -0.94045526 0.8668808; -0.6086891 0.27764758 0.061661094 0.4304493 -0.26440763 0.5692413 -0.41936624 -0.3396712; -0.8256655 -0.26583645 0.46379247 -0.27046546 0.2271158 0.021172568 -0.89212435 0.74854964; 0.29849583 0.26694846 1.0153364 -0.91521335 0.24864979 0.18781434 -1.1007041 0.46872205; -0.3243184 0.27940744 0.7683393 -0.43210885 0.44967622 0.9270794 -1.1782415 0.68745506; 0.41673172 0.4136622 -0.6882764 0.61147124 -1.1496849 -0.36180645 -0.1655861 -0.054610267; -0.42991367 0.520048 -0.25441515 0.6898932 -0.8182926 -0.12463808 0.07750696 -0.09557031; -0.54518694 0.7677194 0.70731115 -0.55656236 0.6088457 0.06417401 -0.49074352 0.65325236], bias = Float32[0.29408672; 0.29185304; 0.13861631; 0.32033718; 0.34290504; 0.11718275; 0.013919995; 0.9827918; 0.38398263; 0.15379696; 0.0056347945; 0.35484645; 0.43214762; 0.06966555; -0.0126404; 0.32342923; 0.8469381; 1.1288179; 1.1546003; 0.7710421; 0.6879667; 1.2347335; 0.6379173; 0.92381287; 0.4784154; 0.029403841; 0.3104948; 0.61724293; 0.6293773; 0.033141293; -0.11907344; 0.87753654;;]), classifier = (weight = Float32[-1.4426415 0.7631253 1.250123 1.269124 -0.9368042 0.12214048 -0.26564062 1.2165569], bias = Float32[-0.5670843;;])), (lstm_cell = (rng = Random.Xoshiro(0x2026f555c226bf09, 0x8a6bb764b93cadda, 0x5ba3c10439600514, 0x446f763658f71987),), classifier = NamedTuple()))
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

