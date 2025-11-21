# # Training a PINN on 2D PDE

# In this tutorial we will go over using a PINN to solve 2D PDEs. We will be using the
# system from [NeuralPDE Tutorials](https://docs.sciml.ai/NeuralPDE/stable/tutorials/gpu/).
# However, we will be using our custom loss function and use nested AD capabilities of
# Lux.jl.

# This is a demonstration of Lux.jl. For serious use cases of PINNs, please refer to
# the package: [NeuralPDE.jl](https://github.com/SciML/NeuralPDE.jl).

# ## Package Imports

using Lux,
    Optimisers,
    Random,
    Printf,
    Statistics,
    MLUtils,
    OnlineStats,
    CairoMakie,
    Reactant,
    Enzyme

const xdev = reactant_device(; force=true)
const cdev = cpu_device()
nothing #hide

# ## Problem Definition

# Since Lux supports efficient nested AD upto 2nd order, we will rewrite the problem
# with first order derivatives, so that we can compute the gradients of the loss using
# 2nd order AD.

# ## Define the Neural Networks

# All the networks take 3 input variables and output a scalar value. Here, we will define
# a wrapper over the 3 networks, so that we can train them using
# [`Training.TrainState`](@ref).

struct PINN{M} <: AbstractLuxWrapperLayer{:model}
    model::M
end

function PINN(; hidden_dims::Int=32)
    return PINN(
        Chain(
            Dense(3 => hidden_dims, tanh),
            Dense(hidden_dims => hidden_dims, tanh),
            Dense(hidden_dims => hidden_dims, tanh),
            Dense(hidden_dims => 1),
        ),
    )
end
nothing #hide

# ## Define the Loss Functions

# We will define a custom loss function to compute the loss using 2nd order AD.
# For that, first we'll need to define the derivatives of our model:

function ∂u_∂t(model::StatefulLuxLayer, xyt::AbstractArray)
    return Enzyme.gradient(Enzyme.Reverse, sum ∘ model, xyt)[1][3, :]
end

function ∂u_∂x(model::StatefulLuxLayer, xyt::AbstractArray)
    return Enzyme.gradient(Enzyme.Reverse, sum ∘ model, xyt)[1][1, :]
end

function ∂u_∂y(model::StatefulLuxLayer, xyt::AbstractArray)
    return Enzyme.gradient(Enzyme.Reverse, sum ∘ model, xyt)[1][2, :]
end

function ∂²u_∂x²(model::StatefulLuxLayer, xyt::AbstractArray)
    return Enzyme.gradient(Enzyme.Reverse, sum ∘ ∂u_∂x, Enzyme.Const(model), xyt)[2][1, :]
end

function ∂²u_∂y²(model::StatefulLuxLayer, xyt::AbstractArray)
    return Enzyme.gradient(Enzyme.Reverse, sum ∘ ∂u_∂y, Enzyme.Const(model), xyt)[2][2, :]
end
nothing #hide

# We will use the following loss function

function physics_informed_loss_function(model::StatefulLuxLayer, xyt::AbstractArray)
    return mean(abs2, ∂u_∂t(model, xyt) .- ∂²u_∂x²(model, xyt) .- ∂²u_∂y²(model, xyt))
end
nothing #hide

# Additionally, we need to compute the loss with respect to the boundary conditions.

function mse_loss_function(
    model::StatefulLuxLayer, target::AbstractArray, xyt::AbstractArray
)
    return MSELoss()(model(xyt), target)
end

function loss_function(model, ps, st, (xyt, target_data, xyt_bc, target_bc))
    smodel = StatefulLuxLayer(model, ps, st)
    physics_loss = physics_informed_loss_function(smodel, xyt)
    data_loss = mse_loss_function(smodel, target_data, xyt)
    bc_loss = mse_loss_function(smodel, target_bc, xyt_bc)
    loss = physics_loss + data_loss + bc_loss
    return loss, smodel.st, (; physics_loss, data_loss, bc_loss)
end
nothing #hide

# ## Generate the Data

# We will generate some random data to train the model on. We will take data on a square
# spatial and temporal domain $x \in [0, 2]$, $y \in [0, 2]$, and $t \in [0, 2]$. Typically,
# you want to be smarter about the sampling process, but for the sake of simplicity, we will
# skip that.

analytical_solution(x, y, t) = @. exp(x + y) * cos(x + y + 4t)
analytical_solution(xyt) = analytical_solution(xyt[1, :], xyt[2, :], xyt[3, :])
nothing #hide
#-

grid_len = 16

grid = range(0.0f0, 2.0f0; length=grid_len)
xyt = stack([[elem...] for elem in vec(collect(Iterators.product(grid, grid, grid)))])

target_data = reshape(analytical_solution(xyt), 1, :)

bc_len = 512

x = collect(range(0.0f0, 2.0f0; length=bc_len))
y = collect(range(0.0f0, 2.0f0; length=bc_len))
t = collect(range(0.0f0, 2.0f0; length=bc_len))

xyt_bc = hcat(
    stack((x, y, zeros(Float32, bc_len)); dims=1),
    stack((zeros(Float32, bc_len), y, t); dims=1),
    stack((ones(Float32, bc_len) .* 2, y, t); dims=1),
    stack((x, zeros(Float32, bc_len), t); dims=1),
    stack((x, ones(Float32, bc_len) .* 2, t); dims=1),
)
target_bc = reshape(analytical_solution(xyt_bc), 1, :)

min_target_bc, max_target_bc = extrema(target_bc)
min_data, max_data = extrema(target_data)
min_pde_val, max_pde_val = min(min_data, min_target_bc), max(max_data, max_target_bc)

xyt = (xyt .- minimum(xyt)) ./ (maximum(xyt) .- minimum(xyt))
xyt_bc = (xyt_bc .- minimum(xyt_bc)) ./ (maximum(xyt_bc) .- minimum(xyt_bc))
target_bc = (target_bc .- min_pde_val) ./ (max_pde_val - min_pde_val)
target_data = (target_data .- min_pde_val) ./ (max_pde_val - min_pde_val)
nothing #hide

# ## Training

function train_model(
    xyt,
    target_data,
    xyt_bc,
    target_bc;
    seed::Int=0,
    maxiters::Int=50000,
    hidden_dims::Int=128,
)
    rng = Random.default_rng()
    Random.seed!(rng, seed)

    pinn = PINN(; hidden_dims)
    ps, st = Lux.setup(rng, pinn) |> xdev

    bc_dataloader =
        DataLoader((xyt_bc, target_bc); batchsize=128, shuffle=true, partial=false) |> xdev
    pde_dataloader =
        DataLoader((xyt, target_data); batchsize=128, shuffle=true, partial=false) |> xdev

    train_state = Training.TrainState(pinn, ps, st, Adam(0.005f0))

    lr = i -> i < 5000 ? 0.005f0 : (i < 10000 ? 0.0005f0 : 0.00005f0)

    total_loss_tracker, physics_loss_tracker, data_loss_tracker, bc_loss_tracker = ntuple(
        _ -> OnlineStats.CircBuff(Float32, 32; rev=true), 4
    )

    iter = 1
    for ((xyt_batch, target_data_batch), (xyt_bc_batch, target_bc_batch)) in
        zip(Iterators.cycle(pde_dataloader), Iterators.cycle(bc_dataloader))
        Optimisers.adjust!(train_state, lr(iter))

        _, loss, stats, train_state = Training.single_train_step!(
            AutoEnzyme(),
            loss_function,
            (xyt_batch, target_data_batch, xyt_bc_batch, target_bc_batch),
            train_state;
            return_gradients=Val(false),
        )

        fit!(total_loss_tracker, Float32(loss))
        fit!(physics_loss_tracker, Float32(stats.physics_loss))
        fit!(data_loss_tracker, Float32(stats.data_loss))
        fit!(bc_loss_tracker, Float32(stats.bc_loss))

        mean_loss = mean(OnlineStats.value(total_loss_tracker))
        mean_physics_loss = mean(OnlineStats.value(physics_loss_tracker))
        mean_data_loss = mean(OnlineStats.value(data_loss_tracker))
        mean_bc_loss = mean(OnlineStats.value(bc_loss_tracker))

        isnan(loss) && throw(ArgumentError("NaN Loss Detected"))

        if iter % 1000 == 1 || iter == maxiters
            @printf(
                "Iteration: [%6d/%6d] \t Loss: %.9f (%.9f) \t Physics Loss: %.9f \
                 (%.9f) \t Data Loss: %.9f (%.9f) \t BC \
                 Loss: %.9f (%.9f)\n",
                iter,
                maxiters,
                loss,
                mean_loss,
                stats.physics_loss,
                mean_physics_loss,
                stats.data_loss,
                mean_data_loss,
                stats.bc_loss,
                mean_bc_loss
            )
        end

        iter += 1
        iter ≥ maxiters && break
    end

    return StatefulLuxLayer(pinn, cdev(train_state.parameters), cdev(train_state.states))
end

trained_model = train_model(xyt, target_data, xyt_bc, target_bc)
nothing #hide

# ## Visualizing the Results
ts, xs, ys = 0.0f0:0.05f0:2.0f0, 0.0f0:0.02f0:2.0f0, 0.0f0:0.02f0:2.0f0
grid = stack([[elem...] for elem in vec(collect(Iterators.product(xs, ys, ts)))])

u_real = reshape(analytical_solution(grid), length(xs), length(ys), length(ts))

grid_normalized = (grid .- minimum(grid)) ./ (maximum(grid) .- minimum(grid))
u_pred = reshape(trained_model(grid_normalized), length(xs), length(ys), length(ts))
u_pred = u_pred .* (max_pde_val - min_pde_val) .+ min_pde_val

begin
    fig = Figure()
    ax = CairoMakie.Axis(fig[1, 1]; xlabel="x", ylabel="y")
    errs = [abs.(u_pred[:, :, i] .- u_real[:, :, i]) for i in 1:length(ts)]
    Colorbar(fig[1, 2]; limits=extrema(stack(errs)))

    CairoMakie.record(fig, "pinn_nested_ad.gif", 1:length(ts); framerate=10) do i
        ax.title = "Abs. Predictor Error | Time: $(ts[i])"
        err = errs[i]
        contour!(ax, xs, ys, err; levels=10, linewidth=2)
        heatmap!(ax, xs, ys, err)
        return fig
    end

    fig
end
nothing #hide
