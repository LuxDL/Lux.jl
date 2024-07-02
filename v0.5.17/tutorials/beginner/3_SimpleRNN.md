


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
                                 sequence_length, 1)
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
((lstm_cell = (weight_i = Float32[-0.86320233 -0.42152035; -0.25324312 -0.75576216; 0.8021475 0.8817252; -0.053329453 0.077659145; -0.1508745 -0.6216561; 0.2921899 0.553875; -0.8266717 0.06464647; 0.084893204 0.060549855; -0.04010294 0.089694805; -0.6592514 0.45209923; 1.1444914 -0.028924467; -0.010410459 0.08980673; -0.10553986 0.24765718; -0.84632397 0.3270066; -0.13181216 -0.28064734; 1.2198714 0.090763725; 0.6755018 -0.6307463; 0.13312562 -0.34829143; 0.4773499 -0.3292663; -0.56122065 0.5170372; 0.58521163 -0.8365467; 0.10655391 0.29521418; 0.8631929 -0.64146024; 0.94397634 0.61126643; -1.2221366 -0.10457709; 0.48708835 0.6466612; 1.0575026 0.67803645; -0.46889886 -0.17536584; 0.72430855 -0.7102103; -0.33265966 0.74639624; -0.2124043 0.6146093; 0.608692 -0.3235226], weight_h = Float32[-0.5113818 -0.051648322 0.2793631 -0.2556458 0.30353603 0.004445685 -0.6939351 0.554755; -0.64524204 0.23564486 -0.06926523 0.64133954 0.38699406 0.31304237 0.13068216 -0.05238549; 0.019921266 0.061315 -0.028669287 0.55480146 -0.746999 0.32265636 -0.6288975 -0.17625023; 0.036403764 -0.27772304 0.8828412 -0.09211041 0.9098435 0.07370045 0.27666336 0.9057811; -0.36025333 0.47491452 0.7458994 0.3074252 0.40262002 0.64021134 -0.3382286 -0.079512745; -0.05906309 -0.33973357 0.33357143 -0.09787208 -0.29181525 -0.222774 -0.49764094 0.043787055; -0.78115916 0.36964753 0.69402224 0.49061564 -0.7993257 0.6381027 0.032236446 -0.63011825; 0.6801192 0.30117223 1.006014 -1.3221166 0.8341015 0.1512931 -0.69941115 1.1529077; -0.13969113 0.474676 0.3981385 -0.5086975 0.4671462 0.673302 0.19902012 0.64464533; -0.41716906 -0.1883017 -0.37945804 -0.014355233 0.24595913 0.0019149594 -0.7506363 0.73670894; -0.14318435 0.5491039 -0.07616947 0.52661264 -0.52986276 0.29071787 -0.26360014 -0.38784415; 0.05671739 0.87803817 0.28378057 -0.10480074 0.8263358 0.59851205 0.3922794 0.4975072; 0.6521741 0.4887229 0.37160265 -0.07114367 0.9206704 0.19096594 -0.6374569 0.44147092; -0.103103966 0.44033647 0.08369991 0.07975464 0.60683435 -0.011747584 -0.13371862 -0.427441; -0.8374503 0.3235005 0.093146466 -0.40554014 -0.2533164 0.8053308 -0.36305323 0.41066787; -0.64595336 0.7877761 0.41382396 1.0418617 -0.091598675 0.8186434 -0.11033947 0.7571849; 0.15321378 -0.6346652 -0.050692637 -0.43474022 -0.35237432 -0.2101065 -0.15971346 0.18070698; -0.43624127 0.2264266 0.20010705 0.71229714 0.3982651 -0.35274932 -0.36507836 0.6819742; -0.5506648 0.7073321 0.094918005 -0.43505073 -0.31369025 0.7374135 0.052997213 0.5975672; -0.7744704 0.339518 -0.16813527 0.500455 -0.69613606 -0.15335473 -0.20049383 -0.1970297; -0.27457952 -0.6667214 0.40499112 -0.64034575 -0.17947614 0.26678616 0.24945422 -0.11277084; -0.7226498 -0.4256362 0.5286779 -0.48049536 -0.16528538 0.026302746 -0.4396535 0.224092; 0.48702767 -0.20146427 -0.66195947 0.29758316 0.34124878 0.06556962 -0.52232695 -0.40717873; -0.40840903 0.35276115 0.20293148 0.36272028 -0.32073808 -0.20065929 -0.74001426 -0.05357915; -0.6385222 0.55719197 0.26229507 -0.46464023 0.6041466 0.43666562 -0.9277107 0.86972964; -0.60920084 0.2811506 0.0640318 0.4303862 -0.26953796 0.56912756 -0.41876924 -0.34197295; -0.8203295 -0.2693611 0.4666494 -0.27377614 0.22461148 0.015289095 -0.886279 0.7484246; 0.30364302 0.2673333 1.0115478 -0.9064344 0.24143262 0.17974338 -1.087922 0.46403572; -0.33819917 0.2824836 0.77169013 -0.4335958 0.45038226 0.9223788 -1.1850656 0.6929358; 0.42469513 0.40669483 -0.6935719 0.6002282 -1.1341732 -0.3647024 -0.17497678 -0.06595013; -0.43512067 0.5188995 -0.24291913 0.688907 -0.8202339 -0.14056629 0.07620188 -0.09708129; -0.5426144 0.76465213 0.70336765 -0.5508476 0.6114614 0.05709072 -0.48327443 0.6527065], bias = Float32[0.29601267; 0.29281056; 0.13649684; 0.32005206; 0.3459972; 0.11511716; 0.013026948; 0.9843024; 0.3864554; 0.15327987; 0.004573738; 0.3542593; 0.4305473; 0.06513069; -0.012064427; 0.3135674; 0.84907395; 1.1271285; 1.1533811; 0.7693958; 0.6889004; 1.2408274; 0.6443962; 0.9233864; 0.4792578; 0.03980642; 0.30369517; 0.61361533; 0.6318478; 0.026863636; -0.11981831; 0.8741794;;]), classifier = (weight = Float32[-1.4424986 0.77166164 1.2472411 1.2649313 -0.9399781 0.117188595 -0.2675722 1.2194527], bias = Float32[-0.55609345;;])), (lstm_cell = (rng = Random.TaskLocalRNG(),), classifier = NamedTuple()))
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

