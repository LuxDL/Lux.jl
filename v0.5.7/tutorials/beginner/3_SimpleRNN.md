


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
((lstm_cell = (weight_i = Float32[-0.8626278 -0.41592368; -0.25446105 -0.75462365; 0.8128315 0.88716793; -0.06357051 0.07564833; -0.14268655 -0.6229416; 0.28473032 0.5514695; -0.83771795 0.07850092; 0.07513294 0.045395594; -0.04100079 0.06923543; -0.6648314 0.452717; 1.1765442 -0.014142505; -0.018047899 0.08731555; -0.110071845 0.24631812; -0.8531736 0.3215939; -0.12699758 -0.26869693; 1.212553 0.09928748; 0.68222666 -0.62769336; 0.12764964 -0.34269774; 0.46679214 -0.32282275; -0.5496983 0.5174812; 0.5831997 -0.83844095; 0.09214563 0.29141894; 0.87278605 -0.6394539; 0.9628667 0.62561435; -1.2522432 -0.10795662; 0.47696868 0.640907; 1.0547856 0.6938462; -0.44210935 -0.15535457; 0.7435911 -0.7010842; -0.34495583 0.729875; -0.21818225 0.60597426; 0.6293672 -0.2951409], weight_h = Float32[-0.51364714 -0.05578296 0.33204323 -0.26356763 0.29335508 0.016487608 -0.67677575 0.54635787; -0.66003317 0.21277228 -0.057907283 0.6270258 0.37851062 0.30923784 0.13063046 -0.04769953; 0.013024257 0.06291374 -0.029871099 0.5646917 -0.7358513 0.31580368 -0.6411012 -0.15206581; 0.039295636 -0.26435396 0.88875157 -0.096749626 0.91130435 0.10403215 0.28320494 0.90724957; -0.37714145 0.46859592 0.7855764 0.30686334 0.40041205 0.6383841 -0.33742797 -0.07942251; -0.060038876 -0.33404046 0.3294067 -0.09318368 -0.29307327 -0.21619241 -0.4881579 0.02902906; -0.7927202 0.3707477 0.6903769 0.52686346 -0.8146131 0.637902 0.019571833 -0.6269248; 0.68901116 0.31119397 1.0859879 -1.306856 0.80355906 0.17665099 -0.7296482 1.1351727; -0.16972464 0.4742745 0.42393687 -0.50196016 0.4553746 0.6855853 0.16665694 0.63149196; -0.4205013 -0.18469375 -0.38307643 -0.006336947 0.23867144 -0.0015613375 -0.7519529 0.7385645; -0.15288973 0.55291086 -0.025671437 0.5288222 -0.5157082 0.29755712 -0.27074474 -0.37197554; 0.058643807 0.88335884 0.2903481 -0.11169035 0.8293076 0.65355784 0.39232588 0.49945393; 0.6506198 0.48768967 0.3754531 -0.07058448 0.92171216 0.16788778 -0.65766287 0.44181967; -0.101398684 0.44513804 0.08479506 0.09742963 0.6033722 -0.012511805 -0.14792629 -0.41848135; -0.8297913 0.3295706 0.08228881 -0.39249668 -0.26507357 0.8011432 -0.37523428 0.41138554; -0.6321523 0.7853601 0.406431 1.0354829 -0.09071835 0.8167014 -0.10001287 0.75440663; 0.14699107 -0.6419354 -0.057768553 -0.4420654 -0.35464463 -0.22630964 -0.16486607 0.17174612; -0.44230527 0.2316889 0.1947993 0.71840924 0.39593485 -0.35616115 -0.36234123 0.6581687; -0.55715 0.708434 0.09339809 -0.42750734 -0.29995808 0.7436798 0.047389906 0.6043874; -0.7770801 0.34042162 -0.16749433 0.50637263 -0.6976446 -0.11512696 -0.1997876 -0.19615929; -0.2674204 -0.6649819 0.3939066 -0.64482194 -0.16483717 0.26282805 0.25014567 -0.12064427; -0.724341 -0.43070123 0.5292423 -0.4802637 -0.13813113 0.03197459 -0.44396168 0.21682903; 0.48673692 -0.18235992 -0.66328454 0.2561593 0.3394808 0.071271226 -0.52100515 -0.41527888; -0.4229447 0.35938126 0.21926698 0.37847078 -0.28841496 -0.18341048 -0.75240993 0.049960833; -0.66132665 0.5518514 0.27382618 -0.44785246 0.59094054 0.43799815 -0.97677827 0.866056; -0.604376 0.26605633 0.072824284 0.42968953 -0.26185107 0.563743 -0.41445968 -0.2996388; -0.81880116 -0.27497867 0.46247625 -0.28390515 0.26089105 0.012612368 -0.8949988 0.7527846; 0.25983834 0.27344295 1.0180728 -0.90857095 0.2551823 0.18786003 -1.135301 0.50819397; -0.3612125 0.27338442 0.7703502 -0.4197131 0.4481407 0.90105337 -1.1872345 0.70767; 0.4450851 0.39042565 -0.6904083 0.64788514 -1.213742 -0.37766755 -0.16215889 -0.061446257; -0.42713532 0.51351595 -0.24785319 0.740743 -0.81545275 -0.12007029 0.09050944 -0.08291307; -0.55131274 0.743675 0.6778393 -0.5318342 0.6502497 0.0591964 -0.49830234 0.63854116], bias = Float32[0.28776008; 0.2681508; 0.1448803; 0.32496646; 0.3423872; 0.122132525; 0.009292153; 0.9703578; 0.37824273; 0.15793608; 0.010978148; 0.35735637; 0.43179154; 0.06795716; -0.013576793; 0.31620887; 0.8424471; 1.1322421; 1.1597546; 0.7686245; 0.6888201; 1.232903; 0.64798766; 0.90879995; 0.47441033; 0.047123976; 0.30493706; 0.6254554; 0.63222575; 0.013876199; -0.118034035; 0.8814841;;]), classifier = (weight = Float32[-1.4337968 0.75605035 1.2362951 1.2632648 -0.9322072 0.11818381 -0.2616262 1.2195352], bias = Float32[-0.6498585;;])), (lstm_cell = (rng = Random.Xoshiro(0x2026f555c226bf09, 0x8a6bb764b93cadda, 0x5ba3c10439600514, 0x446f763658f71987),), classifier = NamedTuple()))
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

