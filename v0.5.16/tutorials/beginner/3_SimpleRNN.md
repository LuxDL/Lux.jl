


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
((lstm_cell = (weight_i = Float32[-0.8620089 -0.41853648; -0.25425282 -0.7545706; 0.8135519 0.8832782; -0.060338385 0.07563762; -0.14370936 -0.6216143; 0.28728515 0.5527795; -0.8345037 0.07474743; 0.06714141 0.051648337; -0.040795315 0.074335694; -0.6628605 0.4523476; 1.1673665 -0.019850638; -0.018012749 0.08730051; -0.10780404 0.24529265; -0.85095984 0.31948057; -0.12851311 -0.27114844; 1.2129608 0.10256768; 0.6802149 -0.62811935; 0.12961593 -0.34477472; 0.47247398 -0.32515934; -0.5512536 0.51761746; 0.58253294 -0.8382177; 0.102347404 0.29349357; 0.87028277 -0.64016306; 0.9485218 0.6252608; -1.2921907 -0.08946486; 0.47764525 0.6423714; 1.0552509 0.6744807; -0.44701168 -0.1480827; 0.74074775 -0.69640756; -0.3419854 0.72623336; -0.21836936 0.6096679; 0.60784197 -0.287415], weight_h = Float32[-0.51910883 -0.055643905 0.2966153 -0.26214767 0.29564992 0.010900508 -0.7069726 0.54798514; -0.6551096 0.2215173 -0.06364299 0.63241744 0.38242918 0.31057426 0.13007772 -0.04372909; 0.016535643 0.06317757 -0.02936801 0.5615001 -0.737726 0.31474555 -0.63767374 -0.14888214; 0.03608441 -0.268631 0.8877468 -0.09395952 0.91181046 0.094645284 0.27852094 0.908274; -0.38106546 0.4690836 0.77703816 0.3163626 0.40105674 0.6354495 -0.34032187 -0.07877857; -0.06098731 -0.33555984 0.3315739 -0.09470686 -0.2895848 -0.21966073 -0.4935742 0.04034542; -0.792021 0.37630078 0.69210744 0.5169532 -0.81143755 0.63824904 0.023596684 -0.618115; 0.6823058 0.30702263 1.0721989 -1.3061023 0.8075969 0.16418035 -0.7135167 1.1426469; -0.17045252 0.47300306 0.41753003 -0.49836564 0.45797512 0.679886 0.17756055 0.63413954; -0.41967058 -0.18584134 -0.3820737 -0.008951106 0.2419606 -0.0007196036 -0.74814147 0.7575475; -0.15078823 0.55398023 -0.04295832 0.52591234 -0.51436013 0.29506615 -0.2695049 -0.36593273; 0.05475289 0.88412905 0.28964463 -0.108952574 0.8292029 0.64185685 0.3890052 0.5007001; 0.6477512 0.4900813 0.37241763 -0.070095435 0.9222879 0.16513735 -0.6942045 0.44239783; -0.098606735 0.4429966 0.08801899 0.09102096 0.60566545 -0.0122444965 -0.1428194 -0.3985202; -0.8255224 0.32566392 0.077545084 -0.39704135 -0.26350433 0.798624 -0.36966038 0.41907957; -0.65011144 0.78817064 0.4152389 1.0824411 -0.079003155 0.8125998 -0.09658827 0.77532876; 0.14888267 -0.63961285 -0.05017922 -0.43976527 -0.3539764 -0.21647105 -0.16354546 0.17316757; -0.440382 0.2298504 0.19733523 0.7162954 0.3954864 -0.35551053 -0.36219817 0.67115724; -0.555159 0.70887566 0.094367355 -0.43001968 -0.3157508 0.7414322 0.049378958 0.60504514; -0.7761272 0.34012574 -0.16733961 0.50499284 -0.6976079 -0.13546272 -0.20009944 -0.1961385; -0.26975068 -0.6645243 0.39684525 -0.6438509 -0.16678137 0.26498696 0.2484658 -0.12822121; -0.7269162 -0.42674792 0.53289497 -0.47798657 -0.14061752 0.028734304 -0.4428491 0.2176939; 0.48658478 -0.17773522 -0.66297275 0.25987366 0.34140033 0.07213418 -0.5209446 -0.42179283; -0.4197796 0.3570052 0.22319946 0.37691635 -0.30970475 -0.18807559 -0.75079346 0.12978391; -0.66801226 0.5511755 0.26793498 -0.44330925 0.590821 0.43509203 -0.95880884 0.8662249; -0.60441536 0.2717929 0.07012442 0.42810482 -0.26488125 0.56447846 -0.41575825 -0.29899806; -0.81896496 -0.2722756 0.46441892 -0.27936497 0.24949789 0.011598398 -0.89144814 0.7559221; 0.25500032 0.27360117 1.0182021 -0.9004374 0.24993248 0.18184632 -1.1280048 0.489584; -0.37728494 0.2740606 0.7662588 -0.41493964 0.44663203 0.8985368 -1.1893995 0.71082795; 0.44888377 0.3875531 -0.6938306 0.63191164 -1.23364 -0.38024858 -0.15086943 -0.056560412; -0.43044573 0.5174612 -0.25907812 0.7171823 -0.8180811 -0.13094276 0.086953975 -0.081010565; -0.5598472 0.74389684 0.6799009 -0.50880444 0.6498257 0.055247765 -0.48648596 0.64562654], bias = Float32[0.28900507; 0.2775703; 0.1408117; 0.32411122; 0.3426988; 0.11992713; 0.01275555; 0.9697849; 0.37834862; 0.1564164; 0.010501348; 0.3575873; 0.43275282; 0.06716698; -0.0139572155; 0.3253667; 0.84440535; 1.130351; 1.1576358; 0.76860523; 0.68949145; 1.2371621; 0.6569105; 0.9117666; 0.47177517; 0.056506153; 0.30499962; 0.6225114; 0.6323346; 0.012518905; -0.11795158; 0.87804556;;]), classifier = (weight = Float32[-1.4301277 0.7656281 1.2388804 1.2600452 -0.93159395 0.11652704 -0.26576817 1.2253661], bias = Float32[-0.67965573;;])), (lstm_cell = (rng = Random.TaskLocalRNG(),), classifier = NamedTuple()))
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

