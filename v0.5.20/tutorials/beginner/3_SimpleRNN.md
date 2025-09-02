


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
    anticlockwise_spirals = [reshape(
                                 d[1][:, (sequence_length + 1):end], :, sequence_length, 1)
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


To understand more about container layers, please look at [Container Layer](../../manual/interface#Container-Layer).


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
    return SpiralClassifier(
        LSTMCell(in_dims => hidden_dims), Dense(hidden_dims => out_dims, sigmoid))
end
```


```
Main.var"##225".SpiralClassifier
```


We can use default Lux blocks – `Recurrence(LSTMCell(in_dims => hidden_dims)` – instead of defining the following. But let's still do it for the sake of it.


Now we need to define the behavior of the Classifier when it is invoked.


```julia
function (s::SpiralClassifier)(
        x::AbstractArray{T, 3}, ps::NamedTuple, st::NamedTuple) where {T}
    # First we will have to run the sequence through the LSTM Cell
    # The first call to LSTM Cell will create the initial hidden state
    # See that the parameters and states are automatically populated into a field called
    # `lstm_cell` We use `eachslice` to get the elements in the sequence without copying,
    # and `Iterators.peel` to split out the first element for LSTM initialization.
    x_init, x_rest = Iterators.peel(Lux._eachslice(x, Val(2)))
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

matches(y_pred, y_true) = sum((y_pred .> 0.5f0) .== y_true)
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
((lstm_cell = (weight_i = Float32[-0.861691 -0.4182207; -0.2534238 -0.7553057; 0.8104275 0.8834548; -0.05779058 0.07526984; -0.1449393 -0.6218852; 0.28877485 0.55152285; -0.8315963 0.0716649; 0.07415046 0.052163385; -0.03985321 0.07746917; -0.6615334 0.45192182; 1.1539111 -0.021582697; -0.0135255875 0.087450475; -0.10740759 0.24583218; -0.8499632 0.32141298; -0.12987587 -0.27720964; 1.215614 0.102995746; 0.6789927 -0.62850034; 0.13114709 -0.34574214; 0.47810355 -0.32646126; -0.5563547 0.5167148; 0.5833233 -0.83786947; 0.09745401 0.2940957; 0.8681477 -0.6395339; 0.94705 0.62477183; -1.2660133 -0.09737401; 0.47938448 0.6427358; 1.0556724 0.67625046; -0.44330177 -0.15993161; 0.7404913 -0.69981307; -0.33939204 0.7333553; -0.21735701 0.61067647; 0.6115292 -0.29619157], weight_h = Float32[-0.5135542 -0.05505858 0.29031095 -0.26330513 0.2972485 0.010044145 -0.7003636 0.549339; -0.65270287 0.22708312 -0.06520428 0.6355265 0.3826406 0.31160903 0.13021088 -0.04576894; 0.017062953 0.061652027 -0.03475262 0.55943924 -0.74034005 0.3194556 -0.6347099 -0.15827842; 0.038060382 -0.2706904 0.8870732 -0.09504072 0.9122202 0.08916883 0.27846625 0.9081613; -0.38086236 0.4703868 0.7745141 0.30408764 0.4014207 0.6383025 -0.34298974 -0.07876707; -0.05935357 -0.33706856 0.33302698 -0.09654483 -0.28802755 -0.21984726 -0.49628916 0.04536251; -0.7875048 0.37222743 0.68972015 0.50813323 -0.8076542 0.63741857 0.025849445 -0.62178034; 0.68739927 0.31143934 1.0489346 -1.3099712 0.81200886 0.16673173 -0.7133407 1.1447277; -0.16088001 0.47356653 0.41243777 -0.5020397 0.4601529 0.68002987 0.18259698 0.63655436; -0.41858256 -0.186889 -0.38146302 -0.010774676 0.24228387 4.3636417f-5 -0.7527611 0.7540056; -0.1491005 0.55198985 -0.054624382 0.5262682 -0.5192912 0.2946343 -0.26760003 -0.3727697; 0.05627829 0.882433 0.28831017 -0.10863705 0.8289834 0.62905157 0.39014623 0.50031704; 0.64954865 0.49173477 0.37243232 -0.07091506 0.9222959 0.17408922 -0.6671275 0.44247082; -0.100772545 0.44317493 0.086660326 0.08662948 0.6078874 -0.012509038 -0.14191332 -0.39917576; -0.8310517 0.3246099 0.08276351 -0.4007236 -0.26017565 0.8083104 -0.3667361 0.41763607; -0.6478505 0.791077 0.4148092 1.0630631 -0.07919441 0.816253 -0.10326125 0.7761577; 0.15013467 -0.63865215 -0.052173954 -0.43877155 -0.3534717 -0.21952254 -0.16274467 0.17538542; -0.4389596 0.22857489 0.19825092 0.715007 0.39539817 -0.3506417 -0.3631293 0.6740575; -0.55382264 0.7082592 0.094241716 -0.4315956 -0.31772858 0.74031013 0.05020278 0.60167617; -0.7754548 0.34086958 -0.1665936 0.50413275 -0.6966237 -0.14120428 -0.19939272 -0.19614455; -0.26994193 -0.6661032 0.39873323 -0.6447515 -0.16734253 0.26363784 0.24708948 -0.12591603; -0.7249698 -0.4268812 0.5322784 -0.47799748 -0.14230555 0.028700469 -0.44120052 0.21902718; 0.48578244 -0.19156514 -0.66008246 0.2789406 0.340799 0.06965717 -0.52243805 -0.41922775; -0.41920134 0.35566157 0.22095598 0.37509674 -0.32002163 -0.18875791 -0.7484064 0.1063529; -0.6554268 0.5525158 0.26753566 -0.45217475 0.59631246 0.43605366 -0.9519897 0.86767507; -0.60530937 0.27492666 0.06657031 0.42891943 -0.26589188 0.5655492 -0.41454226 -0.30669665; -0.81960905 -0.27383074 0.46298066 -0.2753147 0.23802862 0.0133398855 -0.88868475 0.7490592; 0.2736079 0.270502 1.0156398 -0.90567195 0.24688129 0.18223347 -1.1143669 0.4827593; -0.36455956 0.2760421 0.76833695 -0.42929295 0.44834697 0.9054292 -1.1855813 0.70684254; 0.43880218 0.39531568 -0.6929712 0.6180985 -1.1990793 -0.3748288 -0.16175674 -0.056446537; -0.4315063 0.5165631 -0.2606194 0.7097815 -0.8186648 -0.13287683 0.08549568 -0.08508994; -0.55338424 0.7501024 0.6854312 -0.53507024 0.6450654 0.05795212 -0.48815984 0.6468947], bias = Float32[0.29039928; 0.28323886; 0.13914776; 0.32401916; 0.34332654; 0.11856343; 0.011219669; 0.9751394; 0.38017258; 0.15514173; 0.0085724145; 0.35719457; 0.43293506; 0.06804374; -0.013570616; 0.32436007; 0.84533805; 1.1290916; 1.1562148; 0.76950735; 0.6886157; 1.236971; 0.6476744; 0.9111713; 0.4746491; 0.05102166; 0.302431; 0.61919534; 0.6322582; 0.019794017; -0.11864128; 0.8786004;;]), classifier = (weight = Float32[-1.433567 0.7618647 1.2404872 1.2611187 -0.9340738 0.11740283 -0.26427472 1.2315588], bias = Float32[-0.62346315;;])), (lstm_cell = (rng = Random.TaskLocalRNG(),), classifier = NamedTuple()))
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

