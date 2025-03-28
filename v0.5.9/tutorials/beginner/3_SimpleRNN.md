


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
((lstm_cell = (weight_i = Float32[-0.85978055 -0.41630626; -0.2529885 -0.7538623; 0.81250966 0.88673604; -0.058587857 0.076216936; -0.14210005 -0.624262; 0.28747422 0.55229795; -0.8352045 0.07359677; 0.07563126 0.04401956; -0.041436646 0.07176763; -0.66201067 0.45153716; 1.1363187 -0.016388893; -0.011349765 0.0874168; -0.10418178 0.2463673; -0.8499276 0.32237393; -0.12950866 -0.26923934; 1.2124414 0.101897165; 0.6798435 -0.6306412; 0.12981139 -0.34557617; 0.46454096 -0.32528055; -0.5534416 0.5158125; 0.58267146 -0.8401678; 0.09481208 0.2904434; 0.87014896 -0.639614; 0.9576984 0.6232367; -1.2491046 -0.11089584; 0.48347387 0.64061254; 1.0542707 0.6797693; -0.45670268 -0.16698952; 0.7305422 -0.70614564; -0.34088132 0.73956275; -0.21792974 0.6079274; 0.62237036 -0.3063709], weight_h = Float32[-0.50724685 -0.05390771 0.2958498 -0.26549655 0.2955138 0.015682572 -0.6942552 0.54741704; -0.65627736 0.22635667 -0.061439555 0.6318757 0.38138473 0.31214717 0.12912847 -0.046008393; 0.013906986 0.06179228 -0.04774771 0.5639543 -0.73678875 0.3217796 -0.6397077 -0.17133223; 0.039758347 -0.26934043 0.88474214 -0.09848435 0.9111811 0.09445173 0.28076905 0.906808; -0.38335156 0.46891758 0.7755234 0.2933912 0.40003884 0.640429 -0.35064825 -0.08118398; -0.060282916 -0.3360941 0.33203998 -0.09528128 -0.2917854 -0.21858701 -0.49409744 0.042172775; -0.78938085 0.37180233 0.68741506 0.51237226 -0.81138873 0.6393538 0.024612006 -0.6309071; 0.69289374 0.31136018 1.063362 -1.3116267 0.8095192 0.17189638 -0.718987 1.1366651; -0.16008155 0.4755967 0.42339894 -0.506769 0.4560648 0.68492544 0.17481847 0.6320695; -0.41831484 -0.18702482 -0.38097852 -0.010696677 0.24117036 0.00023362534 -0.75075984 0.7548947; -0.15178272 0.552869 -0.042566575 0.52790666 -0.51495725 0.29685152 -0.27149385 -0.3787326; 0.058971956 0.8798312 0.28753185 -0.11331635 0.8292891 0.6368215 0.39324012 0.49970236; 0.6510019 0.49187842 0.37279582 -0.07213205 0.9221449 0.17489137 -0.6369767 0.44199932; -0.102264896 0.44364733 0.088059895 0.087112375 0.6082056 -0.012808944 -0.14204411 -0.40467963; -0.83229405 0.32830215 0.080875084 -0.39802212 -0.26110414 0.79816616 -0.37085432 0.4078067; -0.63543373 0.7900475 0.40788177 1.0534296 -0.07838338 0.81488067 -0.09982296 0.76286626; 0.14819823 -0.6391268 -0.050798204 -0.43859228 -0.35407132 -0.2259635 -0.1625312 0.1733997; -0.43912613 0.22917266 0.19762404 0.715348 0.39812842 -0.3532642 -0.36466068 0.6686518; -0.5549032 0.7086399 0.09439546 -0.43049437 -0.31346425 0.7414041 0.04920624 0.6012293; -0.775204 0.3411627 -0.16508283 0.5030441 -0.6956852 -0.13604826 -0.1995024 -0.19443409; -0.2698716 -0.66699123 0.3988231 -0.64756423 -0.16477048 0.2609193 0.24826151 -0.12816377; -0.7256247 -0.42747718 0.5357168 -0.47652945 -0.14527348 0.029365191 -0.441973 0.2149163; 0.4855953 -0.1910421 -0.66170305 0.2742723 0.33814767 0.06969285 -0.5216369 -0.40915176; -0.42031088 0.35611022 0.22346868 0.37437323 -0.30508918 -0.18809955 -0.74819326 0.026089145; -0.65169126 0.55349725 0.26967886 -0.45675373 0.59402144 0.4398431 -0.9627721 0.86597496; -0.60636467 0.2745433 0.065667674 0.42973104 -0.26001447 0.56690234 -0.41589674 -0.30873555; -0.8233568 -0.27377313 0.45724925 -0.2660457 0.23472974 0.016846543 -0.8949597 0.74047214; 0.27769053 0.2712685 1.0170205 -0.9110697 0.24962741 0.18506216 -1.1224097 0.47957698; -0.34653068 0.27558103 0.76924866 -0.4312104 0.4480843 0.9111946 -1.1861646 0.6982482; 0.43348032 0.39914072 -0.68625164 0.6296665 -1.1739285 -0.37161893 -0.1696267 -0.05660714; -0.42883432 0.51637554 -0.2577299 0.7284277 -0.81567186 -0.11861173 0.0835436 -0.08666104; -0.54578775 0.75417566 0.6916801 -0.552192 0.6445847 0.06325732 -0.49891302 0.6448584], bias = Float32[0.28994334; 0.2810259; 0.13970943; 0.32358548; 0.3416332; 0.11966308; 0.010165458; 0.97365695; 0.3797141; 0.15514743; 0.009293875; 0.3566059; 0.43289018; 0.06811749; -0.012532298; 0.3211174; 0.84502006; 1.1299758; 1.1573519; 0.7702506; 0.68872434; 1.2355454; 0.64263636; 0.91062397; 0.4759365; 0.03647172; 0.30300158; 0.62020326; 0.62890804; 0.02263809; -0.11812414; 0.88300854;;]), classifier = (weight = Float32[-1.4372109 0.747038 1.2411357 1.2648417 -0.9338774 0.11988332 -0.26103795 1.2282157], bias = Float32[-0.6306009;;])), (lstm_cell = (rng = Random.Xoshiro(0x2026f555c226bf09, 0x8a6bb764b93cadda, 0x5ba3c10439600514, 0x446f763658f71987),), classifier = NamedTuple()))
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

