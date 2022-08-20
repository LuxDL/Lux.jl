# # Training a Simple LSTM

# In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:

# 1. Create custom Lux models.
# 2. Become familiar with the Lux recurrent neural network API.
# 3. Training using Optimisers.jl and Zygote.jl.

# ## Package Imports
using Lux
using Pkg #hide
Pkg.activate(joinpath(dirname(pathof(Lux)), "..", "examples")) #hide
using MLUtils, Optimisers, Zygote, NNlib, Random, Statistics

# ## Dataset

# We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a `MLUtils.DataLoader`. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.

function get_dataloaders(; dataset_size=1000, sequence_length=50)
    ## Create the spirals
    data = [MLUtils.Datasets.make_spiral(sequence_length) for _ in 1:dataset_size]
    ## Get the labels
    labels = vcat(repeat([0.0f0], dataset_size ÷ 2), repeat([1.0f0], dataset_size ÷ 2))
    clockwise_spirals = [reshape(d[1][:, 1:sequence_length], :, sequence_length, 1)
                         for d in data[1:(dataset_size ÷ 2)]]
    anticlockwise_spirals = [reshape(d[1][:, (sequence_length + 1):end], :, sequence_length,
                                     1) for d in data[((dataset_size ÷ 2) + 1):end]]
    x_data = Float32.(cat(clockwise_spirals..., anticlockwise_spirals...; dims=3))
    ## Split the dataset
    (x_train, y_train), (x_val, y_val) = splitobs((x_data, labels); at=0.8, shuffle=true)
    ## Create DataLoaders
    return (
            ## Use DataLoader to automatically minibatch and shuffle the data
            DataLoader(collect.((x_train, y_train)); batchsize=128, shuffle=true),
            ## Don't shuffle the validation data
            DataLoader(collect.((x_val, y_val)); batchsize=128, shuffle=false))
end

# ## Creating a Classifier

# We will be extending the `Lux.AbstractExplicitContainerLayer` type for our custom model since it will contain a lstm block and a classifier head.

# We pass the fieldnames `lstm_cell` and `classifier` to the type to ensure that the parameters and states are automatically populated and we don't have to define [`Lux.initialparameters`](@ref) and [`Lux.initialstates`](@ref).

struct SpiralClassifier{L, C} <:
       Lux.AbstractExplicitContainerLayer{(:lstm_cell, :classifier)}
    lstm_cell::L
    classifier::C
end

# We won't define the model from scratch but rather use the [`Lux.LSTMCell`](@ref) and [`Lux.Dense`](@ref).

function SpiralClassifier(in_dims, hidden_dims, out_dims)
    return SpiralClassifier(LSTMCell(in_dims => hidden_dims),
                            Dense(hidden_dims => out_dims, sigmoid))
end

# Now we need to define the behavior of the Classifier when it is invoked.

function (s::SpiralClassifier)(x::AbstractArray{T, 3}, ps::NamedTuple,
                               st::NamedTuple) where {T}
    ## First we will have to run the sequence through the LSTM Cell
    ## The first call to LSTM Cell will create the initial hidden state
    ## See that the parameters and states are automatically populated into a field called `lstm_cell`
    ## We use `eachslice` to get the elements in the sequence without copying,
    ## and `Iterators.peel` to split out the first element for LSTM initialization.
    x_init, x_rest = Iterators.peel(eachslice(x; dims=2))
    (y, carry), st_lstm = s.lstm_cell(x_init, ps.lstm_cell, st.lstm_cell)
    ## Now that we have the hidden state and memory in `carry` we will pass the input and `carry` jointly
    for x in x_rest
        (y, carry), st_lstm = s.lstm_cell((x, carry), ps.lstm_cell, st_lstm)
    end
    ## After running through the sequence we will pass the output through the classifier
    y, st_classifier = s.classifier(y, ps.classifier, st.classifier)
    ## Finally remember to create the updated state
    st = merge(st, (classifier=st_classifier, lstm_cell=st_lstm))
    return vec(y), st
end

# ## Defining Accuracy, Loss and Optimiser

# Now let's define the binarycrossentropy loss. Typically it is recommended to use `logitbinarycrossentropy` since it is more numerically stable, but for the sake of simplicity we will use `binarycrossentropy`.

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

# Finally lets create an optimiser given the model parameters.

function create_optimiser(ps)
    opt = Optimisers.ADAM(0.01f0)
    return Optimisers.setup(opt, ps)
end

# ## Training the Model

function main()
    ## Get the dataloaders
    (train_loader, val_loader) = get_dataloaders()

    ## Create the model
    model = SpiralClassifier(2, 8, 1)
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    ps, st = Lux.setup(rng, model)

    ## Create the optimiser
    opt_state = create_optimiser(ps)

    for epoch in 1:25
        ## Train the model
        for (x, y) in train_loader
            (loss, y_pred, st), back = pullback(p -> compute_loss(x, y, model, p, st), ps)
            gs = back((one(loss), nothing, nothing))[1]
            opt_state, ps = Optimisers.update(opt_state, ps, gs)

            println("Epoch [$epoch]: Loss $loss")
        end

        ## Validate the model
        st_ = Lux.testmode(st)
        for (x, y) in val_loader
            (loss, y_pred, st_) = compute_loss(x, y, model, ps, st_)
            acc = accuracy(y_pred, y)
            println("Validation: Loss $loss Accuracy $acc")
        end
    end
end

main()
