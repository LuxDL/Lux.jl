# # Training a Simple LSTM

# In this tutorial we will go over using a recurrent neural network to classify clockwise
# and anticlockwise spirals. By the end of this tutorial you will be able to:

# 1. Create custom Lux models.
# 2. Become familiar with the Lux recurrent neural network API.
# 3. Training using Optimisers.jl and Zygote.jl.

# ## Package Imports

using ADTypes, Lux, JLD2, MLUtils, Optimisers, Printf, Reactant, Random

# ## Dataset

# We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise
# spirals. Using this data we will create a `MLUtils.DataLoader`. Our dataloader will give
# us sequences of size 2 × seq_len × batch_size and we need to predict a binary value
# whether the sequence is clockwise or anticlockwise.

function get_dataloaders(; dataset_size=1000, sequence_length=50)
    ## Create the spirals
    data = [MLUtils.Datasets.make_spiral(sequence_length) for _ in 1:dataset_size]
    ## Get the labels
    labels = vcat(repeat([0.0f0], dataset_size ÷ 2), repeat([1.0f0], dataset_size ÷ 2))
    clockwise_spirals = [reshape(d[1][:, 1:sequence_length], :, sequence_length, 1)
                         for d in data[1:(dataset_size ÷ 2)]]
    anticlockwise_spirals = [reshape(
                                 d[1][:, (sequence_length + 1):end], :, sequence_length, 1)
                             for d in data[((dataset_size ÷ 2) + 1):end]]
    x_data = Float32.(cat(clockwise_spirals..., anticlockwise_spirals...; dims=3))
    ## Split the dataset
    (x_train, y_train), (x_val, y_val) = splitobs((x_data, labels); at=0.8, shuffle=true)
    ## Create DataLoaders
    return (
        ## Use DataLoader to automatically minibatch and shuffle the data
        DataLoader(
            collect.((x_train, y_train)); batchsize=128, shuffle=true, partial=false),
        ## Don't shuffle the validation data
        DataLoader(collect.((x_val, y_val)); batchsize=128, shuffle=false, partial=false)
    )
end

# ## Creating a Classifier

# We will be extending the `Lux.AbstractLuxContainerLayer` type for our custom model
# since it will contain a lstm block and a classifier head.

# We pass the fieldnames `lstm_cell` and `classifier` to the type to ensure that the
# parameters and states are automatically populated and we don't have to define
# `Lux.initialparameters` and `Lux.initialstates`.

# To understand more about container layers, please look at
# [Container Layer](@ref Container-Layer).

struct SpiralClassifier{L, C} <: AbstractLuxContainerLayer{(:lstm_cell, :classifier)}
    lstm_cell::L
    classifier::C
end

# We won't define the model from scratch but rather use the [`Lux.LSTMCell`](@ref) and
# [`Lux.Dense`](@ref).

function SpiralClassifier(in_dims, hidden_dims, out_dims)
    return SpiralClassifier(
        LSTMCell(in_dims => hidden_dims), Dense(hidden_dims => out_dims, sigmoid))
end

# We can use default Lux blocks -- `Recurrence(LSTMCell(in_dims => hidden_dims)` -- instead
# of defining the following. But let's still do it for the sake of it.

# Now we need to define the behavior of the Classifier when it is invoked.

function (s::SpiralClassifier)(
        x::AbstractArray{T, 3}, ps::NamedTuple, st::NamedTuple) where {T}
    ## First we will have to run the sequence through the LSTM Cell
    ## The first call to LSTM Cell will create the initial hidden state
    ## See that the parameters and states are automatically populated into a field called
    ## `lstm_cell` We use `eachslice` to get the elements in the sequence without copying,
    ## and `Iterators.peel` to split out the first element for LSTM initialization.
    x_init, x_rest = Iterators.peel(LuxOps.eachslice(x, Val(2)))
    (y, carry), st_lstm = s.lstm_cell(x_init, ps.lstm_cell, st.lstm_cell)
    ## Now that we have the hidden state and memory in `carry` we will pass the input and
    ## `carry` jointly
    for x in x_rest
        (y, carry), st_lstm = s.lstm_cell((x, carry), ps.lstm_cell, st_lstm)
    end
    ## After running through the sequence we will pass the output through the classifier
    y, st_classifier = s.classifier(y, ps.classifier, st.classifier)
    ## Finally remember to create the updated state
    st = merge(st, (classifier=st_classifier, lstm_cell=st_lstm))
    return vec(y), st
end

# ## Using the `@compact` API

# We can also define the model using the [`Lux.@compact`](@ref) API, which is a more concise
# way of defining models. This macro automatically handles the boilerplate code for you and
# as such we recommend this way of defining custom layers

function SpiralClassifierCompact(in_dims, hidden_dims, out_dims)
    lstm_cell = LSTMCell(in_dims => hidden_dims)
    classifier = Dense(hidden_dims => out_dims, sigmoid)
    return @compact(; lstm_cell, classifier) do x::AbstractArray{T, 3} where {T}
        x_init, x_rest = Iterators.peel(LuxOps.eachslice(x, Val(2)))
        y, carry = lstm_cell(x_init)
        for x in x_rest
            y, carry = lstm_cell((x, carry))
        end
        @return vec(classifier(y))
    end
end

# ## Defining Accuracy, Loss and Optimiser

# Now let's define the binarycrossentropy loss. Typically it is recommended to use
# `logitbinarycrossentropy` since it is more numerically stable, but for the sake of
# simplicity we will use `binarycrossentropy`.
const lossfn = BinaryCrossEntropyLoss()

function compute_loss(model, ps, st, (x, y))
    ŷ, st_ = model(x, ps, st)
    loss = lossfn(ŷ, y)
    return loss, st_, (; y_pred=ŷ)
end

matches(y_pred, y_true) = sum((y_pred .> 0.5f0) .== y_true)
accuracy(y_pred, y_true) = matches(y_pred, y_true) / length(y_pred)

# ## Training the Model

function main(model_type)
    dev = reactant_device()
    cdev = cpu_device()

    ## Get the dataloaders
    train_loader, val_loader = get_dataloaders() |> dev

    ## Create the model
    model = model_type(2, 8, 1)
    ps, st = Lux.setup(Random.default_rng(), model) |> dev

    train_state = Training.TrainState(model, ps, st, Adam(0.01f0))
    model_compiled = if dev isa ReactantDevice
        @compile model(first(train_loader)[1], ps, Lux.testmode(st))
    else
        model
    end
    ad = dev isa ReactantDevice ? AutoEnzyme() : AutoZygote()

    for epoch in 1:25
        ## Train the model
        total_loss = 0.0f0
        total_samples = 0
        for (x, y) in train_loader
            (_, loss, _, train_state) = Training.single_train_step!(
                ad, lossfn, (x, y), train_state
            )
            total_loss += loss * length(y)
            total_samples += length(y)
        end
        @printf "Epoch [%3d]: Loss %4.5f\n" epoch (total_loss/total_samples)

        ## Validate the model
        total_acc = 0.0f0
        total_loss = 0.0f0
        total_samples = 0

        st_ = Lux.testmode(train_state.states)
        for (x, y) in val_loader
            ŷ, st_ = model_compiled(x, train_state.parameters, st_)
            ŷ, y = cdev(ŷ), cdev(y)
            total_acc += accuracy(ŷ, y) * length(y)
            total_loss += lossfn(ŷ, y) * length(y)
            total_samples += length(y)
        end

        @printf "Validation:\tLoss %4.5f\tAccuracy %4.5f\n" (total_loss/total_samples) (total_acc/total_samples)
    end

    return (train_state.parameters, train_state.states) |> cpu_device()
end

ps_trained, st_trained = main(SpiralClassifier)
nothing #hide

# We can also train the compact model with the exact same code!

ps_trained2, st_trained2 = main(SpiralClassifierCompact)
nothing #hide

# ## Saving the Model

# We can save the model using JLD2 (and any other serialization library of your choice)
# Note that we transfer the model to CPU before saving. Additionally, we recommend that
# you don't save the model struct and only save the parameters and states.

@save "trained_model.jld2" ps_trained st_trained

# Let's try loading the model

@load "trained_model.jld2" ps_trained st_trained
