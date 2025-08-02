


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
((lstm_cell = (weight_i = Float32[-0.8611205 -0.4184025; -0.25388315 -0.75473046; 0.8097058 0.88559943; -0.05805987 0.07744343; -0.14519015 -0.62324804; 0.28840083 0.55315816; -0.8342873 0.06994996; 0.08107819 0.04888144; -0.041523717 0.077374674; -0.66194326 0.45191556; 1.1356302 -0.018663991; -0.013045557 0.08804029; -0.10505265 0.24658449; -0.84958416 0.32413346; -0.12976064 -0.27232093; 1.2122597 0.09078445; 0.6797865 -0.6296126; 0.13030127 -0.3455682; 0.46609274 -0.32556316; -0.5543991 0.5159944; 0.5837504 -0.8380326; 0.09926474 0.291142; 0.8687048 -0.63997173; 0.9521395 0.6193421; -1.2461451 -0.10603744; 0.4826795 0.64281964; 1.0544599 0.67902696; -0.46000457 -0.16634299; 0.72990733 -0.7055795; -0.3393952 0.7400718; -0.21621373 0.6111499; 0.60711795 -0.31201366], weight_h = Float32[-0.50947845 -0.053226467 0.2918976 -0.2601493 0.29725808 0.0131402165 -0.69014287 0.5489027; -0.6531756 0.22662908 -0.06331284 0.6342396 0.3834615 0.3118914 0.13016951 -0.049056724; 0.014137805 0.06319288 -0.039481547 0.56282026 -0.7397722 0.3215091 -0.6382573 -0.1749107; 0.038618114 -0.27168277 0.8829614 -0.09717276 0.90999126 0.089550115 0.2804682 0.9060723; -0.371874 0.4711316 0.77334917 0.30628228 0.4003688 0.6401207 -0.34472108 -0.08072005; -0.06062243 -0.3365798 0.33163494 -0.09545993 -0.29082462 -0.21873598 -0.49442995 0.040744085; -0.7881734 0.37151012 0.6936133 0.5076739 -0.8079327 0.63969857 0.024708519 -0.6289962; 0.68642026 0.3062554 1.0448602 -1.3145298 0.81662333 0.16622697 -0.7182278 1.1417019; -0.15417473 0.47548804 0.41694444 -0.509094 0.4587944 0.68211144 0.18027408 0.6355168; -0.4183943 -0.1868519 -0.3806638 -0.010821198 0.24273893 2.4815343f-5 -0.7531652 0.74253273; -0.15029588 0.55220884 -0.05321938 0.528103 -0.51850027 0.29540926 -0.27034962 -0.38223857; 0.058188137 0.8794132 0.28769365 -0.111318156 0.8287617 0.6293917 0.39243075 0.49992326; 0.65082437 0.4921891 0.37323856 -0.07291194 0.92201596 0.17967512 -0.62819904 0.44259736; -0.1031782 0.44389787 0.08716303 0.08733781 0.60783046 -0.012155324 -0.13775566 -0.41020083; -0.8314855 0.32850164 0.08703067 -0.39858142 -0.25918892 0.80194753 -0.3700432 0.41187152; -0.6493276 0.788747 0.4094063 1.0854172 -0.075068206 0.81333786 -0.10224173 0.77351934; 0.14823915 -0.63874686 -0.050228942 -0.43834707 -0.35332632 -0.21981585 -0.162805 0.17515114; -0.43895176 0.22897539 0.19761512 0.7150527 0.39907655 -0.35302445 -0.36344254 0.6693486; -0.5544548 0.7081951 0.09494743 -0.43098754 -0.31152478 0.7408752 0.049677026 0.60174584; -0.7746319 0.3406368 -0.16561553 0.50091994 -0.6953312 -0.13578945 -0.20048477 -0.19439168; -0.2710844 -0.6663027 0.39942226 -0.6456238 -0.16868138 0.26309726 0.24945831 -0.12112063; -0.72514135 -0.42676476 0.53317064 -0.47884718 -0.14476484 0.029529123 -0.44202828 0.21721984; 0.48621583 -0.19533552 -0.66346925 0.28215516 0.33883703 0.06871395 -0.522055 -0.41090065; -0.41933137 0.35696924 0.2246885 0.37497222 -0.32974064 -0.19065857 -0.7475161 0.05145903; -0.649797 0.5547136 0.26846188 -0.46010575 0.59633774 0.4391669 -0.9546483 0.86696047; -0.6071799 0.27375734 0.06623593 0.43015024 -0.26214883 0.5672999 -0.4188304 -0.32071134; -0.8227413 -0.27061164 0.46242526 -0.26771647 0.23047073 0.015928023 -0.8926456 0.7446127; 0.28007314 0.27183294 1.0168905 -0.91202784 0.24907213 0.18715288 -1.1188052 0.47676438; -0.34350103 0.27902618 0.7702579 -0.42780152 0.447999 0.9157779 -1.1866672 0.6973295; 0.43064776 0.40203622 -0.68873674 0.6292463 -1.1757613 -0.36987734 -0.16982473 -0.05650232; -0.43237823 0.52004147 -0.25498363 0.70423454 -0.81740063 -0.121549696 0.07914718 -0.089094654; -0.54893434 0.75913596 0.6987539 -0.54855204 0.62508535 0.06257067 -0.49752307 0.6477625], bias = Float32[0.2916398; 0.2821655; 0.13999777; 0.32209224; 0.34302342; 0.119044274; 0.012297967; 0.97658414; 0.38210207; 0.15539935; 0.008414747; 0.35615653; 0.43255916; 0.06799398; -0.011818474; 0.32573652; 0.8451628; 1.1298763; 1.1570346; 0.7702907; 0.68870735; 1.2380987; 0.6420919; 0.916017; 0.47721037; 0.033531047; 0.30493796; 0.6212391; 0.6310675; 0.023409106; -0.118108444; 0.8799233;;]), classifier = (weight = Float32[-1.4389681 0.7553157 1.2428886 1.2660484 -0.93516266 0.1198058 -0.26512966 1.2177043], bias = Float32[-0.5857342;;])), (lstm_cell = (rng = Random.Xoshiro(0x2026f555c226bf09, 0x8a6bb764b93cadda, 0x5ba3c10439600514, 0x446f763658f71987),), classifier = NamedTuple()))
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

