# # Using Enzyme to Train Lux Neural Networks

# In this tutorial we will go over using a recurrent neural network to classify clockwise
# and anticlockwise spirals. This is a slightly advanced and verbose tutorial using
# Enzyme's autodiff directly. By the end of this tutorial you will be able to:

# 1. Compute gradients of Lux models using Enzyme.jl.
# 2. Training using Optimisers.jl and Enzyme.jl.

# One thing to note here that Enzyme support for Lux is still under development and there
# might be some rough edges. However, we expect most CPU code to work out of the box, but
# CUDA support is mostly untested.

# ## Package Imports
import Pkg #hide
__DIR = @__DIR__ #hide
pkg_io = open(joinpath(__DIR, "pkg.log"), "w") #hide
Pkg.activate(__DIR; io=pkg_io) #hide
Pkg.instantiate(; io=pkg_io) #hide
Pkg.develop(; path=joinpath(__DIR, "..", ".."), io=pkg_io) #hide
Pkg.precompile(; io=pkg_io) #hide
close(pkg_io) #hide
using ADTypes, Enzyme, Lux, MLUtils, Optimisers, Printf, Random

Enzyme.API.typeWarning!(false)
Enzyme.API.runtimeActivity!(true)

## Dataset

# We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise
# spirals. Using this data we will create a `MLUtils.DataLoader`. Our dataloader will give
# us sequences of size 2 × seq_len × batch_size and we need to predict a binary value
# whether the sequence is clockwise or anticlockwise.

function get_dataloaders(; dataset_size=4000, sequence_length=50)
    ## Create the spirals
    data = [MLUtils.Datasets.make_spiral(sequence_length) for _ in 1:dataset_size]
    ## Get the labels
    labels = vcat(repeat([0.0f0], dataset_size ÷ 2), repeat([1.0f0], dataset_size ÷ 2))
    clockwise_spirals = [reshape(d[1][:, 1:sequence_length], :, sequence_length, 1)
                         for d in data[1:(dataset_size ÷ 2)]]
    anticlockwise_spirals = [reshape(
                                 d[1][:, (sequence_length + 1):end], :, sequence_length, 1)
                             for d in data[((dataset_size ÷ 2) + 1):end]]
    x_data = Float32.(cat(clockwise_spirals..., anticlockwise_spirals...; dims=Val(3)))
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

# In the SimpleRNN tutorial we used a custom model. Here we will do the exact same, so we
# will skip the explanation of the model.

struct SpiralClassifier{L, C} <:
       Lux.AbstractExplicitContainerLayer{(:lstm_cell, :classifier)}
    lstm_cell::L
    classifier::C
end

function SpiralClassifier(in_dims, hidden_dims, out_dims)
    return SpiralClassifier(
        LSTMCell(in_dims => hidden_dims), Dense(hidden_dims => out_dims, sigmoid))
end

function (s::SpiralClassifier)(
        x::AbstractArray{T, 3}, ps::NamedTuple, st::NamedTuple) where {T}
    x_list = Lux._eachslice(x, Val(2))
    x_init = first(x_list)
    (y, carry), st_lstm = s.lstm_cell(x_init, ps.lstm_cell, st.lstm_cell)
    for x in view(x_list, 2:lastindex(x_list))
        (y, carry), st_lstm = s.lstm_cell((x, carry), ps.lstm_cell, st_lstm)
    end
    y, st_classifier = s.classifier(y, ps.classifier, st.classifier)
    st = merge(st, (classifier=st_classifier, lstm_cell=st_lstm))
    return vec(y), st
end

# ## Defining Accuracy, Loss and Optimiser

# Now let's define the binarycrossentropy loss. Typically it is recommended to use
# `logitbinarycrossentropy` since it is more numerically stable, but for the sake of
# simplicity we will use `binarycrossentropy`.

function xlogy(x, y)
    result = x * log(y)
    return ifelse(iszero(x), zero(result), result)
end

function binarycrossentropy(y_pred, y_true)
    y_pred = y_pred .+ eps(eltype(y_pred))
    return sum(@. -xlogy(y_true, y_pred) - xlogy(1 - y_true, 1 - y_pred))
end

function compute_loss(model, ps, st, (x, y))
    y_pred, st = model(x, ps, st)
    return binarycrossentropy(y_pred, y), st
end

matches(y_pred, y_true) = sum((y_pred .> 0.5f0) .== y_true)
accuracy(y_pred, y_true) = matches(y_pred, y_true) / length(y_pred)

# ## Training the Model

# Lux uses states `st` to carry around information about the model, so we need the loss
# function to return multiple outputs (non-scalar). Hence, we will use the
# `Enzyme.autodiff_thunk` API with `ReverseModeSplit`. For more details refer to their
# documentation [here](https://enzyme.mit.edu/julia/dev/).

# function main()
## Get the dataloaders
(train_loader, val_loader) = get_dataloaders()

## Create the model
model = SpiralClassifier(2, 8, 1)
rng = Xoshiro(0)
ps, st = Lux.setup(rng, model)

## Do the autodiff_thunk and cache the results
x_, y_ = first(train_loader)
## Here we are essentially passing in the types of the argument to `compute_loss`. We ≘
## telling enzyme to compute the gradient only wrt `ps`
forward, reverse = Enzyme.autodiff_thunk(
    Enzyme.ReverseSplitWithPrimal, Const{typeof(compute_loss)},
    Active, Const{typeof(model)}, Duplicated{typeof(ps)},
    Const{typeof(st)}, Const{typeof((x_, y_))})

train_state = Lux.Experimental.TrainState(rng, model, Adam(0.01f0); transform_variables=dev)

for epoch in 1:25
    ## Train the model
    for (x, y) in train_loader
        x = x |> dev
        y = y |> dev

        gs, loss, _, train_state = Lux.Experimental.compute_gradients(
            AutoZygote(), compute_loss, (x, y), train_state)
        train_state = Lux.Experimental.apply_gradients(train_state, gs)

        @printf "Epoch [%3d]: Loss %4.5f\n" epoch loss
    end

    ## Validate the model
    st_ = Lux.testmode(train_state.states)
    for (x, y) in val_loader
        x = x |> dev
        y = y |> dev
        loss, st_, ret = compute_loss(model, train_state.parameters, st_, (x, y))
        acc = accuracy(ret.y_pred, y)
        @printf "Validation: Loss %4.5f Accuracy %4.5f\n" loss acc
    end
end

return (train_state.parameters, train_state.states) |> cpu_device()
# end

ps_trained, st_trained = main()
nothing #hide
