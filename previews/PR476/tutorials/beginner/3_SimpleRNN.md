


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
((lstm_cell = (weight_i = Float32[-0.861691 -0.41653624; -0.25256047 -0.7554812; 0.814027 0.88463515; -0.061639905 0.07603954; -0.1421942 -0.62068814; 0.28682333 0.5511059; -0.8374778 0.078011215; 0.06732454 0.052002296; -0.039815675 0.06897186; -0.6631802 0.45137045; 1.1736715 -0.017040225; -0.019031934 0.08789792; -0.110750824 0.24409902; -0.85194415 0.31778815; -0.128812 -0.26891643; 1.2158293 0.10719954; 0.68056136 -0.6281537; 0.1296414 -0.34383577; 0.45728433 -0.32352835; -0.5505671 0.5170883; 0.5819827 -0.83659315; 0.089916974 0.29205862; 0.87150025 -0.63880515; 0.9548461 0.6275347; -1.2535607 -0.09153152; 0.48017707 0.6401966; 1.055061 0.67546463; -0.44653323 -0.14412685; 0.74445695 -0.69406706; -0.34395745 0.71837294; -0.2200745 0.60803634; 0.62582165 -0.2836879], weight_h = Float32[-0.5211804 -0.057081934 0.3008646 -0.2663274 0.29327923 0.0121369045 -0.7191351 0.5461713; -0.65888715 0.22090413 -0.06008428 0.6316774 0.37859565 0.30965862 0.1310721 -0.041109767; 0.014655923 0.061933864 -0.03989087 0.565713 -0.7357016 0.31607085 -0.6408302 -0.14664283; 0.035047486 -0.266939 0.8888683 -0.09292526 0.91149503 0.10306034 0.27775013 0.9079496; -0.3755786 0.46671662 0.78035146 0.3230682 0.40033427 0.6328168 -0.33592156 -0.0792309; -0.05974571 -0.33551404 0.33196655 -0.095065534 -0.28769735 -0.21895938 -0.4968026 0.04180314; -0.79249924 0.37429708 0.6836048 0.5250988 -0.8131575 0.6371162 0.020335093 -0.62118006; 0.685017 0.30806923 1.0983442 -1.2986679 0.7979602 0.16326961 -0.71896493 1.1377648; -0.17750612 0.47267678 0.4277476 -0.4960945 0.45741495 0.6814196 0.17011532 0.63459134; -0.41891444 -0.1863002 -0.38372043 -0.007835649 0.2389385 -0.0019596983 -0.7569937 0.75663465; -0.1518278 0.5527313 -0.035770647 0.527477 -0.5116871 0.29623756 -0.27005935 -0.36398458; 0.053690594 0.88573724 0.29082686 -0.110577434 0.8291134 0.65337306 0.38609728 0.5006009; 0.6474909 0.49318725 0.37458915 -0.07205413 0.9243718 0.15502918 -0.7176214 0.44421083; -0.09525119 0.44270742 0.089086555 0.093809724 0.6056234 -0.017744096 -0.14538848 -0.401416; -0.83590144 0.33041805 0.08404386 -0.3912188 -0.26547137 0.7995141 -0.3731791 0.4232544; -0.6398971 0.7884421 0.4163594 1.0621821 -0.08052285 0.8111028 -0.09590307 0.7661184; 0.14914271 -0.6411643 -0.05531446 -0.4446841 -0.35117126 -0.22907206 -0.16431649 0.16904028; -0.44086462 0.22993799 0.19689928 0.71749496 0.39135018 -0.35691422 -0.36282235 0.66560197; -0.55658233 0.7096531 0.09443857 -0.428536 -0.31911004 0.74291277 0.048237015 0.6053649; -0.7761741 0.3408934 -0.16674441 0.5060598 -0.69739294 -0.12862082 -0.19943228 -0.19542706; -0.26675266 -0.66470605 0.3986857 -0.6480977 -0.15920249 0.26349786 0.24765332 -0.12285606; -0.7267507 -0.42690897 0.536329 -0.47606128 -0.14070089 0.029686028 -0.44220772 0.21762802; 0.48610187 -0.19018152 -0.65650564 0.2480254 0.34221983 0.07253829 -0.52148515 -0.4157713; -0.4193286 0.35692775 0.22324333 0.3771264 -0.2910106 -0.18601933 -0.75173753 0.10247833; -0.66271657 0.5472767 0.26758388 -0.44323957 0.5945882 0.4324803 -0.9614819 0.86989886; -0.6017651 0.26548886 0.06803531 0.42868558 -0.26448375 0.560881 -0.4138074 -0.28761452; -0.8200005 -0.27565113 0.460781 -0.27450344 0.25327918 0.013075481 -0.8943475 0.7491164; 0.25041813 0.2729406 1.0166345 -0.90306795 0.25632372 0.1768002 -1.1398422 0.49887753; -0.384728 0.27495542 0.7664847 -0.4190335 0.44914854 0.8838225 -1.1983415 0.71972555; 0.45831504 0.38019112 -0.6949551 0.641725 -1.2255169 -0.38728303 -0.15479991 -0.067403786; -0.43294019 0.51864326 -0.25993147 0.7458358 -0.8169837 -0.13016613 0.084539734 -0.07777596; -0.5588785 0.7363849 0.6737618 -0.5081695 0.6665653 0.04884485 -0.4930213 0.64358544], bias = Float32[0.28682905; 0.27884328; 0.14455104; 0.32424545; 0.3417299; 0.12073116; 0.01080203; 0.9653216; 0.37687206; 0.1563851; 0.011856654; 0.3576651; 0.43494034; 0.06777566; -0.012026286; 0.32334036; 0.8436563; 1.1303917; 1.1589485; 0.7690281; 0.6897186; 1.2358614; 0.64282006; 0.9074866; 0.47269672; 0.05541127; 0.30533886; 0.6281381; 0.6376241; 0.006893485; -0.11655535; 0.8760433;;]), classifier = (weight = Float32[-1.4298013 0.7489584 1.2335236 1.2646403 -0.935373 0.11427325 -0.2665416 1.2183832], bias = Float32[-0.66864896;;])), (lstm_cell = (rng = Random.Xoshiro(0x2026f555c226bf09, 0x8a6bb764b93cadda, 0x5ba3c10439600514, 0x446f763658f71987, 0x22a21880af5dc689),), classifier = NamedTuple()))
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

