# # MLFlow Lux.jl Integration Guide

# In this tutorial, we fit a linear regression using a neural network and recorded the 
# experimental data into MLFlow.

# ## Package Imports

# We need to use [MLFlowClient.jl](https://github.com/JuliaAI/MLFlowClient.jl) to log experimental data.

using Distributions, Lux, MLUtils, Random, Optimisers, Printf, Zygote, MLFlowClient

# ## Pre-requisites

# Before starting the code below, you should have an MLFlow server running locally.

# ## Dataset

# Generate 100 datapoints from the polynomial $y = x*w + b$.
function synthetic_data(w::Vector{<:Real},b::Real,num_example::Int)
    X = randn(Float32,(num_example,length(w)))
    y = Float32.(X * w .+ b)
    y += rand(Normal(0f0,0.01f0),(size(y)))
    return X',reshape(y,(1,:))
end

true_w = [2,-3.4]
true_b = 4.2
features,labels = synthetic_data(true_w,true_b,100)

train_loader = DataLoader((features,labels),batchsize=10,shuffle=true)

# ## Neural Network

# A simple Dense layer is sufficient to fit this function.

rng = Xoshiro(0)
model = Dense(2=>1)
ps, st = Lux.setup(rng, model)

# ## Loss Function & Optimizer

const mse = MSELoss()
opt = Descent()

# ## Configure MLFlow

# Create MLFlow instance
mlf = MLFlow("http://localhost:5000/api")

# Initiate new experiment
experiment_id = getorcreateexperiment(mlf, "Linear Regression")

# Create a run in the new experiment
exprun = createrun(mlf, experiment_id)

# Log parameters and their values
weight,bias = vec(ps[1]),first(ps[2])
for (idx, w) in enumerate(weight)
    logparam(mlf, exprun, "weight$(idx)", w) # MLFlow only supports string parameter values
end
logparam(mlf,exprun,"bias",bias)

# ## Training
let train_state = Training.TrainState(model, ps, st,opt)
    num_epochs = 3
    for epoch in 1:num_epochs
        for data in train_loader
            (_, loss, _, train_state) = Training.single_train_step!(AutoZygote(), mse, data, train_state)
        end
        epoch_loss = mse(model(features,ps,st)[1],labels)
        @printf "epoch %i, loss %f \n" epoch epoch_loss
        #Log loss in MLFlow
        logmetric(mlf, exprun, "loss", epoch_loss; step=epoch)
    end
end

# complete the experiment
updaterun(mlf, exprun, "FINISHED")

# Now you can check this experiment on the MLFlow dashboard.

# ![MLFlowDashboard](./assets/mlflow.jpg)