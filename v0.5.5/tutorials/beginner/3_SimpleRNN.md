


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
((lstm_cell = (weight_i = Float32[-0.8614019 -0.42128605; -0.2537665 -0.75535643; 0.8070142 0.8834409; -0.05667875 0.076101564; -0.1458845 -0.6250416; 0.28956148 0.55431163; -0.8334061 0.06821155; 0.077941746 0.053303577; -0.041325595 0.080461375; -0.6609663 0.45220098; 1.1479199 -0.022628814; -0.0111318035 0.087910295; -0.10533681 0.24690609; -0.8482719 0.32370177; -0.13006026 -0.27218965; 1.2104505 0.098945744; 0.6789659 -0.6292496; 0.13128059 -0.3469456; 0.47371083 -0.32698414; -0.5578456 0.51704127; 0.5817672 -0.8398508; 0.106921315 0.29382262; 0.8672621 -0.6402486; 0.94154686 0.6188719; -1.2782065 -0.10071971; 0.48618838 0.64445007; 1.0551913 0.67745155; -0.45292997 -0.167972; 0.7386822 -0.7038584; -0.3369602 0.7394637; -0.21593456 0.6136382; 0.6104299 -0.3063524], weight_h = Float32[-0.5115791 -0.05321867 0.290654 -0.2592501 0.2996299 0.010396876 -0.6991235 0.55097914; -0.6504008 0.22913891 -0.06525955 0.6357531 0.3864935 0.31160927 0.13033637 -0.047656868; 0.017701428 0.061580695 -0.031656858 0.55879724 -0.74244714 0.3193232 -0.634148 -0.16656913; 0.037566114 -0.27251616 0.8853197 -0.0954604 0.9112984 0.086392656 0.27880934 0.9075432; -0.37580734 0.47146198 0.77410364 0.3068223 0.40207916 0.6388229 -0.34733388 -0.07880786; -0.061469156 -0.33720848 0.33248985 -0.095860936 -0.28882915 -0.22091016 -0.49649477 0.044526506; -0.78657126 0.37236783 0.6924661 0.5007508 -0.8052185 0.6395196 0.028344756 -0.6249669; 0.68418723 0.3059616 1.0354019 -1.3165207 0.82021576 0.15904339 -0.70802975 1.1447362; -0.15443514 0.47440872 0.40871283 -0.5037755 0.46066 0.67837936 0.1859214 0.63636243; -0.41847444 -0.18736693 -0.38106623 -0.011955811 0.24380276 0.0008490011 -0.7495044 0.75816184; -0.14748918 0.5513659 -0.058697835 0.52675134 -0.52317363 0.29282358 -0.26703343 -0.38016945; 0.05663416 0.8815808 0.28637543 -0.10718533 0.8282111 0.6240551 0.3910264 0.49933955; 0.64910513 0.49485502 0.37221178 -0.069256395 0.9207923 0.18161502 -0.6450938 0.44130152; -0.10156435 0.44291005 0.08860628 0.0840371 0.60854274 -0.011744825 -0.13590276 -0.4039553; -0.8260446 0.3254426 0.080331095 -0.40348035 -0.2574762 0.7974974 -0.36726302 0.4124276; -0.6469862 0.7912826 0.41025865 1.0647285 -0.079577565 0.814877 -0.102369726 0.7699342; 0.14992104 -0.6379534 -0.05149009 -0.43680027 -0.35420522 -0.2165845 -0.16224764 0.17790927; -0.43825155 0.22801845 0.19836064 0.7142968 0.39707616 -0.35510233 -0.3634913 0.67703676; -0.5530237 0.70798725 0.09407748 -0.43292743 -0.34028074 0.73963755 0.051410645 0.60127157; -0.77509403 0.34027618 -0.16719042 0.5029525 -0.6965835 -0.15193757 -0.20008513 -0.1962474; -0.27034363 -0.66566986 0.40201953 -0.6439418 -0.16781974 0.26486278 0.24949718 -0.12325495; -0.7263434 -0.42521173 0.52991647 -0.47896543 -0.13971163 0.027470266 -0.44200772 0.22034967; 0.48642218 -0.19390537 -0.6615584 0.284239 0.3398588 0.0677457 -0.5218113 -0.4121917; -0.4144706 0.35505864 0.20735064 0.37011167 -0.3246396 -0.19576772 -0.7450362 0.017971348; -0.6543157 0.55610365 0.2673456 -0.45223552 0.59543663 0.43854117 -0.95151263 0.86688244; -0.6075056 0.2797216 0.06767027 0.42877623 -0.26643714 0.5686086 -0.41808605 -0.31591117; -0.82123977 -0.27137047 0.46441692 -0.27232435 0.22947986 0.014540715 -0.8899203 0.75091904; 0.28417793 0.27300745 1.0177053 -0.8982737 0.23787904 0.18277161 -1.1087421 0.47519165; -0.35232034 0.2797581 0.77115035 -0.42417344 0.44654685 0.9151805 -1.1867517 0.70174676; 0.43148053 0.40244144 -0.68956846 0.61377007 -1.2048576 -0.36800256 -0.16137813 -0.056288306; -0.43625185 0.5214259 -0.25683185 0.7036643 -0.81945145 -0.1271279 0.07895126 -0.08774804; -0.552387 0.7550493 0.6921838 -0.53766805 0.6265188 0.060002416 -0.48705205 0.6455965], bias = Float32[0.2926866; 0.28551957; 0.1368107; 0.322734; 0.34397486; 0.118158996; 0.012926836; 0.977571; 0.38214546; 0.15460426; 0.006481984; 0.35649735; 0.4320371; 0.067674935; -0.012473635; 0.3225012; 0.8457373; 1.128736; 1.1556671; 0.76928484; 0.6902605; 1.2405854; 0.6437488; 0.9158989; 0.47640425; 0.042163823; 0.30359143; 0.6171688; 0.6322281; 0.024367435; -0.11803894; 0.8776362;;]), classifier = (weight = Float32[-1.4371357 0.7685496 1.2452022 1.2589986 -0.93403137 0.11938751 -0.26822454 1.219812], bias = Float32[-0.61194646;;])), (lstm_cell = (rng = Random.Xoshiro(0x2026f555c226bf09, 0x8a6bb764b93cadda, 0x5ba3c10439600514, 0x446f763658f71987),), classifier = NamedTuple()))
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

