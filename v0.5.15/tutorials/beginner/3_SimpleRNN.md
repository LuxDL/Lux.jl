


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
((lstm_cell = (weight_i = Float32[-0.85937935 -0.4150431; -0.25313777 -0.75460356; 0.8126757 0.8856749; -0.05828349 0.07595368; -0.14415155 -0.62330234; 0.28665146 0.55204505; -0.83489984 0.07246271; 0.08300574 0.041996352; -0.039984625 0.07198565; -0.6625018 0.45160565; 1.1356386 -0.016443733; -0.0104716765 0.08724262; -0.10474415 0.24677937; -0.850654 0.32270947; -0.12907903 -0.2744775; 1.2129916 0.09813542; 0.68101364 -0.62909544; 0.12923393 -0.3447071; 0.4713744 -0.3247732; -0.55320376 0.5160251; 0.58463633 -0.83688515; 0.09654514 0.28988943; 0.8700069 -0.6397288; 0.9646356 0.6226685; -1.2287964 -0.11468172; 0.4823311 0.6405884; 1.0548153 0.6795787; -0.45690405 -0.16687223; 0.72521204 -0.7063486; -0.34046027 0.7425542; -0.21786888 0.6068104; 0.6228871 -0.31017536], weight_h = Float32[-0.50618756 -0.054031175 0.3148837 -0.26502925 0.29483852 0.017393205 -0.6814037 0.5476911; -0.6567909 0.22205849 -0.060930707 0.62963766 0.3805991 0.3114007 0.13013428 -0.04980094; 0.015628573 0.061020046 -0.042631883 0.5629233 -0.73613846 0.32007128 -0.6376184 -0.1739546; 0.040330715 -0.2682697 0.8849189 -0.098369256 0.9112229 0.09725605 0.28171095 0.90678525; -0.3803208 0.4696666 0.77418727 0.2841309 0.40008128 0.6427349 -0.34522802 -0.08079991; -0.060189456 -0.33565614 0.33123398 -0.09464145 -0.2918173 -0.21784447 -0.49014145 0.044809002; -0.7888943 0.370105 0.68694055 0.513359 -0.81084716 0.6396132 0.024177054 -0.6311727; 0.69031775 0.31783533 1.0626878 -1.3132366 0.80795705 0.18226863 -0.73092747 1.1334765; -0.1582662 0.4753833 0.42567003 -0.5068646 0.4550392 0.6866929 0.17420694 0.63158673; -0.41825533 -0.18673639 -0.38127798 -0.010216191 0.2412337 -0.00021153879 -0.75569737 0.7331427; -0.15134934 0.5517893 -0.039432712 0.5272424 -0.5134491 0.29728317 -0.27029666 -0.3801233; 0.06020628 0.8800242 0.28700194 -0.11325079 0.8292954 0.63841635 0.39396128 0.49946457; 0.6525597 0.48930186 0.3720204 -0.07138404 0.9214364 0.17991985 -0.6281747 0.44161826; -0.10367429 0.4448074 0.086575426 0.08900266 0.608564 -0.01135232 -0.14321156 -0.40502185; -0.83418757 0.32698262 0.079659306 -0.3980136 -0.25985292 0.80928355 -0.37017578 0.40662718; -0.63970214 0.7859255 0.4086437 1.0311129 -0.09386958 0.82013834 -0.10781902 0.75937736; 0.14743845 -0.6400276 -0.04926139 -0.4377407 -0.35569137 -0.22390747 -0.16342147 0.17531385; -0.43967733 0.22972801 0.19666795 0.71579087 0.39937627 -0.35029528 -0.36283165 0.6645151; -0.5554135 0.70808285 0.09375954 -0.42961958 -0.3157431 0.74212325 0.04851919 0.601809; -0.7756981 0.3413654 -0.16476852 0.50381935 -0.6958672 -0.1356123 -0.1990605 -0.19545618; -0.27120894 -0.66709745 0.39522728 -0.6452576 -0.1670569 0.2597103 0.2473169 -0.12891883; -0.7240901 -0.42944866 0.53642154 -0.47987458 -0.14483504 0.03023184 -0.4429191 0.21455479; 0.4860816 -0.19685946 -0.6631586 0.28472352 0.33833408 0.06812503 -0.5222625 -0.4129456; -0.42190772 0.35672387 0.24061246 0.37555468 -0.30645388 -0.18739238 -0.74881566 0.0621734; -0.647837 0.5532101 0.27143577 -0.45846468 0.5949159 0.44057178 -0.9619071 0.8657158; -0.60657007 0.2718303 0.06558456 0.4300301 -0.25918078 0.5671259 -0.41496342 -0.31590486; -0.8224501 -0.27390525 0.4579399 -0.26898584 0.23804861 0.016365081 -0.8942019 0.7422755; 0.2819349 0.27000275 1.0172551 -0.9108743 0.25186288 0.18609442 -1.1174644 0.49565554; -0.3403063 0.27659327 0.7712343 -0.42653093 0.44753507 0.91852134 -1.1856426 0.6976799; 0.42978293 0.40146586 -0.6858291 0.6284337 -1.1597385 -0.36790884 -0.16857532 -0.054479152; -0.42711386 0.51338583 -0.26117787 0.72914255 -0.81475025 -0.1189709 0.08657745 -0.08859814; -0.542122 0.7552255 0.69520724 -0.5555086 0.6353271 0.06531222 -0.499149 0.6448855], bias = Float32[0.28974628; 0.27724966; 0.13731323; 0.3239356; 0.3421519; 0.1203946; 0.0098862415; 0.97386575; 0.3799838; 0.155656; 0.00889179; 0.3567742; 0.43193573; 0.06903161; -0.013429086; 0.3136074; 0.8438637; 1.1307398; 1.1580328; 0.7701399; 0.6866516; 1.2348237; 0.65013176; 0.912013; 0.47664374; 0.037071843; 0.30186844; 0.6195354; 0.62804604; 0.025962738; -0.11857459; 0.88430196;;]), classifier = (weight = Float32[-1.4383262 0.7486438 1.2405707 1.2642341 -0.9336931 0.12104358 -0.25893897 1.2274274], bias = Float32[-0.6106102;;])), (lstm_cell = (rng = Random.Xoshiro(0x2026f555c226bf09, 0x8a6bb764b93cadda, 0x5ba3c10439600514, 0x446f763658f71987, 0x22a21880af5dc689),), classifier = NamedTuple()))
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

