


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
((lstm_cell = (weight_i = Float32[-0.86344516 -0.41870427; -0.25297913 -0.7558243; 0.8144164 0.8843971; -0.063257456 0.07654123; -0.14654922 -0.6188375; 0.28592792 0.5501489; -0.8399197 0.07961349; 0.06826201 0.053149328; -0.037406545 0.0729887; -0.6642651 0.4517104; 1.1831771 -0.017400183; -0.018994465 0.08793379; -0.114471756 0.24448013; -0.8536111 0.31492624; -0.12802035 -0.26917955; 1.2204808 0.109427385; 0.68150914 -0.62705964; 0.12861045 -0.34260446; 0.45689693 -0.32316905; -0.5502916 0.5178931; 0.5833815 -0.8349808; 0.09465774 0.29230407; 0.8737256 -0.63912016; 0.958752 0.6272021; -1.2024282 -0.09364044; 0.47733915 0.6413168; 1.05591 0.6732327; -0.4405649 -0.15021472; 0.75680566 -0.6937403; -0.34512678 0.71487045; -0.22296056 0.60889006; 0.6319872 -0.28912938], weight_h = Float32[-0.5180471 -0.058824293 0.32426155 -0.26189458 0.29378015 0.009500588 -0.7171535 0.54613227; -0.6594144 0.21190031 -0.058749456 0.62846136 0.37578917 0.30847117 0.13175164 -0.044431206; 0.014766166 0.06192846 -0.032852974 0.5671907 -0.7383681 0.3127238 -0.64209855 -0.15070418; 0.03616051 -0.2656965 0.88727295 -0.09146333 0.9106873 0.09885901 0.2793744 0.90695536; -0.3611744 0.46875137 0.78286827 0.33951378 0.40161306 0.63291824 -0.32662317 -0.07789746; -0.058923643 -0.33496806 0.3311979 -0.09441694 -0.28602567 -0.21885298 -0.4917314 0.04078317; -0.7936645 0.3743906 0.6863742 0.52666205 -0.8155575 0.63728285 0.020080365 -0.6236311; 0.6846959 0.30410862 1.0966496 -1.3021553 0.8037508 0.16022438 -0.71833956 1.1422355; -0.17088562 0.47078115 0.43149325 -0.49640608 0.45960253 0.67865384 0.17131914 0.6376649; -0.41979447 -0.18559837 -0.38397503 -0.006269302 0.23777601 -0.0029617127 -0.75614786 0.7329256; -0.15077075 0.55157477 -0.031053642 0.52711344 -0.51397514 0.29527843 -0.27019942 -0.36762133; 0.055857133 0.8851735 0.29027715 -0.10897321 0.82916254 0.6481904 0.38639298 0.5003126; 0.64900166 0.48987335 0.3744035 -0.07217887 0.92384773 0.15201879 -0.71498483 0.44440833; -0.09122512 0.44201028 0.0869057 0.09525719 0.60588324 -0.018901287 -0.14370856 -0.4070269; -0.83246815 0.32847908 0.08250055 -0.39345285 -0.2669301 0.7954009 -0.374235 0.4188154; -0.63217044 0.78719634 0.42318398 1.0756495 -0.07333393 0.8096867 -0.094238244 0.7733911; 0.14985573 -0.6423474 -0.055437062 -0.44617325 -0.34898704 -0.22610946 -0.1654768 0.16870031; -0.44237676 0.23110431 0.19558743 0.7188331 0.38983804 -0.35785258 -0.3615557 0.657879; -0.5567712 0.70830625 0.093463786 -0.4278244 -0.31879532 0.74324924 0.04804483 0.60565287; -0.77713066 0.34008646 -0.16781272 0.5061518 -0.6981998 -0.12510367 -0.19982708 -0.1959705; -0.26630718 -0.6634587 0.3967635 -0.64494574 -0.1626759 0.26458082 0.24927106 -0.11704544; -0.7256172 -0.428429 0.5367381 -0.4798884 -0.13351938 0.03087798 -0.44379523 0.22067758; 0.4865594 -0.177903 -0.6600136 0.23564887 0.3445953 0.074458115 -0.5203273 -0.41412735; -0.4174871 0.35725936 0.22962308 0.37965733 -0.2962598 -0.18766446 -0.751782 0.10770624; -0.6535219 0.5454457 0.2656376 -0.44849786 0.6012359 0.43015504 -0.9523602 0.8734635; -0.6010159 0.25975814 0.06937063 0.42865798 -0.26716438 0.5587635 -0.41352782 -0.29486215; -0.8175142 -0.2755088 0.46187386 -0.27946326 0.25678128 0.0107163405 -0.8942178 0.75479233; 0.25449702 0.2724211 1.0187715 -0.9075692 0.25380108 0.17535858 -1.139526 0.4965488; -0.3813355 0.27206054 0.76257867 -0.43351406 0.45444304 0.8753361 -1.1962408 0.72102016; 0.4644498 0.375617 -0.69657505 0.6370705 -1.2419761 -0.38973615 -0.1418343 -0.07221971; -0.43151623 0.51785094 -0.2604513 0.74454784 -0.8174828 -0.13230176 0.088796206 -0.0800793; -0.5548374 0.7375263 0.67690414 -0.51118946 0.6617994 0.04719546 -0.4924524 0.64494824], bias = Float32[0.286578; 0.2705307; 0.1427883; 0.3242584; 0.34392753; 0.12168259; 0.0107903415; 0.9685888; 0.37785813; 0.1573548; 0.0104883555; 0.35745803; 0.43364272; 0.06773867; -0.013566202; 0.32663673; 0.84276485; 1.131494; 1.15947; 0.7682047; 0.6892041; 1.236459; 0.65107226; 0.90746856; 0.47485211; 0.05430249; 0.30560634; 0.6265285; 0.640852; 0.004117163; -0.11648563; 0.8772018;;]), classifier = (weight = Float32[-1.4316508 0.7510565 1.230469 1.2670354 -0.9392783 0.11132709 -0.26543 1.2120007], bias = Float32[-0.64185023;;])), (lstm_cell = (rng = Random.Xoshiro(0x2026f555c226bf09, 0x8a6bb764b93cadda, 0x5ba3c10439600514, 0x446f763658f71987),), classifier = NamedTuple()))
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

