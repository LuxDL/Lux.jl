


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
((lstm_cell = (weight_i = Float32[-0.85811055 -0.41448572; -0.2539219 -0.7548152; 0.80814755 0.8852014; -0.054183163 0.07788212; -0.14214928 -0.6242846; 0.28991985 0.55474854; -0.8339243 0.070523225; 0.08605138 0.046491943; -0.042814568 0.071784884; -0.660093 0.45232776; 1.1229087 -0.01898816; -0.008341092 0.08933561; -0.101063855 0.24730194; -0.8465383 0.32978174; -0.13168882 -0.27279872; 1.2102609 0.09463438; 0.6770903 -0.63203895; 0.13212255 -0.34788764; 0.47011882 -0.3265385; -0.5577269 0.51527804; 0.58229226 -0.8412245; 0.091842756 0.2927797; 0.8669633 -0.641038; 0.9493938 0.6175627; -1.2741721 -0.117681816; 0.48893735 0.6412502; 1.0548533 0.6821888; -0.43790686 -0.17306617; 0.73370355 -0.7062965; -0.33652207 0.75525934; -0.21568196 0.6086928; 0.6083216 -0.31680852], weight_h = Float32[-0.506145 -0.051693656 0.28588122 -0.2665868 0.29597095 0.017783789 -0.68744534 0.5486051; -0.6472835 0.24043377 -0.06566435 0.6419299 0.3860222 0.31387302 0.13038148 -0.04791387; 0.0163881 0.061181556 -0.04704284 0.55956495 -0.73787993 0.325392 -0.6362081 -0.17991146; 0.037692823 -0.2749442 0.88286483 -0.09713857 0.90933734 0.08790637 0.27910262 0.90512586; -0.38894242 0.46932358 0.77034 0.28691104 0.39903533 0.64202905 -0.3556398 -0.082447164; -0.062059745 -0.33743528 0.33323354 -0.09620953 -0.2939799 -0.21944703 -0.5040775 0.038103916; -0.7862043 0.3700605 0.69233155 0.49954873 -0.8067866 0.6423759 0.02971331 -0.633441; 0.6855659 0.31133643 1.0365943 -1.3168285 0.81700075 0.16785982 -0.7136884 1.1371382; -0.15731104 0.4774572 0.41230097 -0.51279145 0.45554206 0.6864484 0.17922805 0.63068813; -0.41814828 -0.18766572 -0.37971455 -0.01343851 0.24442233 0.0034128523 -0.7454914 0.7775584; -0.14979875 0.55267334 -0.061499093 0.5276836 -0.51902336 0.2966208 -0.2700775 -0.38551977; 0.05739611 0.8772089 0.2858052 -0.11269864 0.82722265 0.6244938 0.39331263 0.4977905; 0.650709 0.49482465 0.37371176 -0.074024916 0.9224425 0.18707848 -0.61998606 0.44165263; -0.10970054 0.44331476 0.090209134 0.082588665 0.60558695 -0.009627136 -0.13808294 -0.4109626; -0.8293403 0.3259754 0.08844605 -0.40198556 -0.2580784 0.80532944 -0.36594725 0.40673226; -0.6498996 0.7870176 0.40417138 1.0498546 -0.08453632 0.81673175 -0.105336845 0.7686799; 0.15101893 -0.6363837 -0.05067949 -0.4370776 -0.35346633 -0.22654547 -0.16039173 0.17563795; -0.43674377 0.2271411 0.1999105 0.7131693 0.3984029 -0.34869266 -0.3640205 0.6810394; -0.55359626 0.71058834 0.0953429 -0.43301526 -0.30965006 0.7402936 0.05033046 0.59932774; -0.77361876 0.34160987 -0.16484807 0.5006011 -0.69469476 -0.14305529 -0.19902135 -0.19513686; -0.27153808 -0.6691414 0.40194413 -0.6492313 -0.16346885 0.2593935 0.2471345 -0.12802824; -0.7270814 -0.4234091 0.53075534 -0.47057214 -0.15479752 0.027024549 -0.43991685 0.21213289; 0.48635343 -0.1850162 -0.66378635 0.28585985 0.33916876 0.06686577 -0.52056956 -0.41022602; -0.4172343 0.35554698 0.22246487 0.3684346 -0.33786666 -0.19122955 -0.7443325 0.05194163; -0.6589933 0.5594547 0.27246854 -0.46308112 0.59031796 0.44652298 -0.9676083 0.86323214; -0.6084231 0.285356 0.060314745 0.4306436 -0.2586967 0.57397074 -0.41769812 -0.32414052; -0.82493824 -0.2723713 0.46244434 -0.26470298 0.22451854 0.019682506 -0.8921842 0.7374822; 0.28933838 0.2719345 1.0156977 -0.91367203 0.24672502 0.19082063 -1.1113487 0.4690483; -0.33638737 0.27817005 0.77014154 -0.4342492 0.44605342 0.9258205 -1.1797628 0.6921848; 0.41578898 0.41340786 -0.681928 0.6187951 -1.1533207 -0.35869974 -0.18005803 -0.04740421; -0.4288591 0.5194867 -0.2503616 0.69975436 -0.81635034 -0.11005836 0.0791903 -0.090628475; -0.5457505 0.76441455 0.70232683 -0.56028503 0.6175382 0.069910534 -0.49955222 0.647347], bias = Float32[0.29135272; 0.29597193; 0.1363889; 0.32044342; 0.34100616; 0.11733999; 0.011469639; 0.97589225; 0.38021138; 0.1538259; 0.007868358; 0.35453835; 0.43349844; 0.066837594; -0.011787564; 0.31791043; 0.8476433; 1.1278268; 1.1558653; 0.7709929; 0.68878055; 1.2374315; 0.6526955; 0.91711724; 0.47672397; 0.031512678; 0.3012261; 0.6185851; 0.62897503; 0.03382294; -0.11787574; 0.87901807;;]), classifier = (weight = Float32[-1.4399923 0.749919 1.2481492 1.2661375 -0.9334753 0.12678139 -0.263455 1.2210237], bias = Float32[-0.6036812;;])), (lstm_cell = (rng = Random.Xoshiro(0x2026f555c226bf09, 0x8a6bb764b93cadda, 0x5ba3c10439600514, 0x446f763658f71987),), classifier = NamedTuple()))
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

