


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
((lstm_cell = (weight_i = Float32[-0.8655862 -0.42187297; -0.25244427 -0.75678104; 0.80195165 0.88040805; -0.05574982 0.07496528; -0.14843719 -0.61859053; 0.2915123 0.5519925; -0.83252275 0.06785115; 0.06675246 0.06365236; -0.037010845 0.08608017; -0.6599177 0.45185262; 1.1651539 -0.027411535; -0.014312581 0.08723193; -0.108772054 0.2440024; -0.84755933 0.32433474; -0.13137227 -0.28019196; 1.2263267 0.09690487; 0.6756627 -0.62917; 0.13296463 -0.3476869; 0.47277105 -0.32800245; -0.5593153 0.51745045; 0.5837007 -0.8342119; 0.10130897 0.29915494; 0.8655423 -0.6386502; 0.93847686 0.619266; -1.2302815 -0.08754921; 0.48529127 0.6433655; 1.0566202 0.6758504; -0.44596556 -0.15600635; 0.74627787 -0.6972066; -0.33607164 0.7274695; -0.21934213 0.6132149; 0.6113087 -0.29341093], weight_h = Float32[-0.52086836 -0.056413647 0.28520942 -0.2614573 0.30098814 0.00044524163 -0.7134397 0.5533003; -0.6469778 0.23460148 -0.06744617 0.6410174 0.38355988 0.3116131 0.13172147 -0.04252191; 0.021356815 0.058313232 -0.029806754 0.5573489 -0.7457675 0.31884968 -0.63056594 -0.14802842; 0.033837616 -0.27550247 0.8903307 -0.09021041 0.91318613 0.07973713 0.273564 0.90960884; -0.3700163 0.47257167 0.75519615 0.3220919 0.40367797 0.6345129 -0.33403423 -0.07589591; -0.058066703 -0.33909622 0.33370295 -0.09847896 -0.28802377 -0.22271264 -0.50182074 0.04524217; -0.78199315 0.37093064 0.69252133 0.4998858 -0.803938 0.6340662 0.033948634 -0.6178676; 0.68012726 0.3056609 1.0455848 -1.3072971 0.8182351 0.14649367 -0.69610786 1.1538336; -0.15587227 0.47090137 0.40100592 -0.49576688 0.46873584 0.6699624 0.19588757 0.6471154; -0.41791758 -0.18820752 -0.38164386 -0.01230066 0.24381138 0.00045466036 -0.743942 0.7572768; -0.14471906 0.55029464 -0.069286406 0.5288027 -0.5294577 0.29110956 -0.2632068 -0.37304252; 0.05378918 0.8855823 0.28986916 -0.105321966 0.82912916 0.6153093 0.38627064 0.50120384; 0.6493648 0.49321666 0.37316144 -0.07021891 0.92357266 0.16856322 -0.70993406 0.44397792; -0.09844172 0.4400258 0.08390505 0.082618885 0.6053283 -0.01725526 -0.13516237 -0.41933918; -0.8401718 0.3193096 0.08863155 -0.40424457 -0.25899106 0.804513 -0.36196578 0.42389753; -0.64484286 0.78900707 0.42159045 1.0441612 -0.09481875 0.8187464 -0.10932389 0.7674164; 0.15510492 -0.6358634 -0.055452663 -0.43807226 -0.34949318 -0.21138592 -0.16069624 0.17656034; -0.43749624 0.22677456 0.19977057 0.7136408 0.39141273 -0.35537624 -0.3638575 0.68286574; -0.55184174 0.7083554 0.0949552 -0.43383643 -0.32583052 0.73868066 0.052777525 0.60083497; -0.7751846 0.34042147 -0.16829866 0.50447065 -0.6974138 -0.14505114 -0.19925141 -0.19761539; -0.26947507 -0.6658472 0.4090093 -0.64316183 -0.17013319 0.26831612 0.24674495 -0.11729698; -0.72564095 -0.42490897 0.52709365 -0.47560948 -0.16335687 0.02696283 -0.4395608 0.22390668; 0.48455116 -0.18619327 -0.66072845 0.27463943 0.34420806 0.07142537 -0.5236254 -0.41747585; -0.40930998 0.3537913 0.20040065 0.36665255 -0.30746856 -0.19499502 -0.74461544 0.022612307; -0.6482785 0.54955125 0.25884005 -0.44741896 0.60530126 0.42970783 -0.9262966 0.87352437; -0.6029186 0.27499264 0.06513886 0.42852002 -0.273437 0.5616003 -0.41165748 -0.3084353; -0.8177513 -0.2779916 0.46249726 -0.2738059 0.23394038 0.012400039 -0.8847769 0.7484226; 0.27827382 0.2672669 1.0109088 -0.8947439 0.24452388 0.16722365 -1.1008804 0.48606086; -0.383024 0.2764474 0.7674662 -0.4345019 0.45339796 0.889479 -1.1913383 0.7165747; 0.44869694 0.38785213 -0.6970412 0.60147613 -1.1891447 -0.38093513 -0.1502776 -0.0650007; -0.43516326 0.516401 -0.24320768 0.70713854 -0.8213693 -0.15153915 0.08501632 -0.08583468; -0.5549776 0.74911416 0.68294996 -0.5117191 0.6518249 0.04891605 -0.47734392 0.6533003], bias = Float32[0.29149887; 0.29120907; 0.13788842; 0.3227795; 0.34587914; 0.116139784; 0.011214878; 0.97582567; 0.3820757; 0.1536611; 0.006197558; 0.35799554; 0.43407205; 0.065949924; -0.0146185905; 0.3136091; 0.8483428; 1.1270752; 1.1544144; 0.7688923; 0.6897322; 1.2385069; 0.649241; 0.91602343; 0.47532323; 0.06806798; 0.29816976; 0.6170062; 0.63844705; 0.01496602; -0.11906548; 0.8751029;;]), classifier = (weight = Float32[-1.4321394 0.7625702 1.2385211 1.2601593 -0.9398832 0.112424254 -0.26811096 1.2336565], bias = Float32[-0.61106104;;])), (lstm_cell = (rng = Random.Xoshiro(0x2026f555c226bf09, 0x8a6bb764b93cadda, 0x5ba3c10439600514, 0x446f763658f71987),), classifier = NamedTuple()))
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

