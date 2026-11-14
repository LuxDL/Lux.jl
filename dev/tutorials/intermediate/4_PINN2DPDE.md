---
url: /dev/tutorials/intermediate/4_PINN2DPDE.md
---

# Training a PINN on 2D PDE {#Training-a-PINN-on-2D-PDE}

In this tutorial we will go over using a PINN to solve 2D PDEs. We will be using the system from [NeuralPDE Tutorials](https://docs.sciml.ai/NeuralPDE/stable/tutorials/gpu/). However, we will be using our custom loss function and use nested AD capabilities of Lux.jl.

This is a demonstration of Lux.jl. For serious use cases of PINNs, please refer to the package: [NeuralPDE.jl](https://github.com/SciML/NeuralPDE.jl).

## Package Imports {#Package-Imports}

```julia
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
```

## Problem Definition {#Problem-Definition}

Since Lux supports efficient nested AD upto 2nd order, we will rewrite the problem with first order derivatives, so that we can compute the gradients of the loss using 2nd order AD.

## Define the Neural Networks {#Define-the-Neural-Networks}

All the networks take 3 input variables and output a scalar value. Here, we will define a wrapper over the 3 networks, so that we can train them using [`Training.TrainState`](/api/Lux/utilities#Lux.Training.TrainState).

```julia
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
```

## Define the Loss Functions {#Define-the-Loss-Functions}

We will define a custom loss function to compute the loss using 2nd order AD. For that, first we'll need to define the derivatives of our model:

```julia
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
```

We will use the following loss function

```julia
function physics_informed_loss_function(model::StatefulLuxLayer, xyt::AbstractArray)
    return mean(abs2, ∂u_∂t(model, xyt) .- ∂²u_∂x²(model, xyt) .- ∂²u_∂y²(model, xyt))
end
```

Additionally, we need to compute the loss with respect to the boundary conditions.

```julia
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
```

## Generate the Data {#Generate-the-Data}

We will generate some random data to train the model on. We will take data on a square spatial and temporal domain $x \in \[0, 2]$, $y \in \[0, 2]$, and $t \in \[0, 2]$. Typically, you want to be smarter about the sampling process, but for the sake of simplicity, we will skip that.

```julia
analytical_solution(x, y, t) = @. exp(x + y) * cos(x + y + 4t)
analytical_solution(xyt) = analytical_solution(xyt[1, :], xyt[2, :], xyt[3, :])
```

```julia
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
```

## Training {#Training}

```julia
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
```

```
Iteration: [     1/ 50000] 	 Loss: 20.523933411 (20.523933411) 	 Physics Loss: 16.931318283 (16.931318283) 	 Data Loss: 2.007483006 (2.007483006) 	 BC Loss: 1.585133195 (1.585133195)
Iteration: [  1001/ 50000] 	 Loss: 0.017368630 (0.019241150) 	 Physics Loss: 0.000384361 (0.000523634) 	 Data Loss: 0.005318496 (0.007538575) 	 BC Loss: 0.011665773 (0.011178941)
Iteration: [  2001/ 50000] 	 Loss: 0.015431651 (0.018665710) 	 Physics Loss: 0.001248566 (0.001662084) 	 Data Loss: 0.004322520 (0.006408240) 	 BC Loss: 0.009860564 (0.010595388)
Iteration: [  3001/ 50000] 	 Loss: 0.015749747 (0.015216007) 	 Physics Loss: 0.000569922 (0.001279062) 	 Data Loss: 0.004014886 (0.004232429) 	 BC Loss: 0.011164939 (0.009704515)
Iteration: [  4001/ 50000] 	 Loss: 0.009721044 (0.008719840) 	 Physics Loss: 0.002388357 (0.003385662) 	 Data Loss: 0.003175572 (0.002104991) 	 BC Loss: 0.004157114 (0.003229188)
Iteration: [  5001/ 50000] 	 Loss: 0.006757717 (0.007407871) 	 Physics Loss: 0.002375914 (0.003384324) 	 Data Loss: 0.002151557 (0.001632576) 	 BC Loss: 0.002230246 (0.002390969)
Iteration: [  6001/ 50000] 	 Loss: 0.001092593 (0.001356918) 	 Physics Loss: 0.000297981 (0.000328636) 	 Data Loss: 0.000584401 (0.000765949) 	 BC Loss: 0.000210212 (0.000262334)
Iteration: [  7001/ 50000] 	 Loss: 0.001391995 (0.000973599) 	 Physics Loss: 0.000266460 (0.000310934) 	 Data Loss: 0.001007391 (0.000518549) 	 BC Loss: 0.000118144 (0.000144116)
Iteration: [  8001/ 50000] 	 Loss: 0.000864801 (0.000893241) 	 Physics Loss: 0.000455246 (0.000375047) 	 Data Loss: 0.000318668 (0.000415710) 	 BC Loss: 0.000090887 (0.000102484)
Iteration: [  9001/ 50000] 	 Loss: 0.002029578 (0.002988601) 	 Physics Loss: 0.001025635 (0.002112256) 	 Data Loss: 0.000749859 (0.000513153) 	 BC Loss: 0.000254083 (0.000363192)
Iteration: [ 10001/ 50000] 	 Loss: 0.000606705 (0.000700462) 	 Physics Loss: 0.000213503 (0.000329162) 	 Data Loss: 0.000294203 (0.000307215) 	 BC Loss: 0.000098998 (0.000064085)
Iteration: [ 11001/ 50000] 	 Loss: 0.000366659 (0.000378968) 	 Physics Loss: 0.000138599 (0.000067893) 	 Data Loss: 0.000184541 (0.000273617) 	 BC Loss: 0.000043518 (0.000037459)
Iteration: [ 12001/ 50000] 	 Loss: 0.000263182 (0.000353990) 	 Physics Loss: 0.000048748 (0.000066280) 	 Data Loss: 0.000171280 (0.000250976) 	 BC Loss: 0.000043154 (0.000036734)
Iteration: [ 13001/ 50000] 	 Loss: 0.000308139 (0.000327301) 	 Physics Loss: 0.000063397 (0.000066381) 	 Data Loss: 0.000208884 (0.000227942) 	 BC Loss: 0.000035859 (0.000032979)
Iteration: [ 14001/ 50000] 	 Loss: 0.000383965 (0.000333207) 	 Physics Loss: 0.000068670 (0.000068110) 	 Data Loss: 0.000283375 (0.000236465) 	 BC Loss: 0.000031920 (0.000028633)
Iteration: [ 15001/ 50000] 	 Loss: 0.000238627 (0.000289024) 	 Physics Loss: 0.000043933 (0.000057306) 	 Data Loss: 0.000162911 (0.000200758) 	 BC Loss: 0.000031783 (0.000030959)
Iteration: [ 16001/ 50000] 	 Loss: 0.000220579 (0.000291252) 	 Physics Loss: 0.000047240 (0.000060969) 	 Data Loss: 0.000142773 (0.000201675) 	 BC Loss: 0.000030566 (0.000028608)
Iteration: [ 17001/ 50000] 	 Loss: 0.000424045 (0.000299802) 	 Physics Loss: 0.000094147 (0.000068041) 	 Data Loss: 0.000306943 (0.000203182) 	 BC Loss: 0.000022955 (0.000028578)
Iteration: [ 18001/ 50000] 	 Loss: 0.000213268 (0.000282862) 	 Physics Loss: 0.000041610 (0.000060157) 	 Data Loss: 0.000142541 (0.000195869) 	 BC Loss: 0.000029117 (0.000026836)
Iteration: [ 19001/ 50000] 	 Loss: 0.000216196 (0.000271350) 	 Physics Loss: 0.000061876 (0.000049561) 	 Data Loss: 0.000134699 (0.000197611) 	 BC Loss: 0.000019621 (0.000024178)
Iteration: [ 20001/ 50000] 	 Loss: 0.000293526 (0.000254105) 	 Physics Loss: 0.000038409 (0.000049744) 	 Data Loss: 0.000235016 (0.000181028) 	 BC Loss: 0.000020101 (0.000023333)
Iteration: [ 21001/ 50000] 	 Loss: 0.000303403 (0.000240493) 	 Physics Loss: 0.000058332 (0.000047394) 	 Data Loss: 0.000224041 (0.000170352) 	 BC Loss: 0.000021029 (0.000022747)
Iteration: [ 22001/ 50000] 	 Loss: 0.000167927 (0.000238713) 	 Physics Loss: 0.000032361 (0.000048484) 	 Data Loss: 0.000109742 (0.000167914) 	 BC Loss: 0.000025824 (0.000022316)
Iteration: [ 23001/ 50000] 	 Loss: 0.000228312 (0.000254367) 	 Physics Loss: 0.000036888 (0.000055099) 	 Data Loss: 0.000168826 (0.000177017) 	 BC Loss: 0.000022598 (0.000022251)
Iteration: [ 24001/ 50000] 	 Loss: 0.000297164 (0.000254442) 	 Physics Loss: 0.000058808 (0.000057595) 	 Data Loss: 0.000218741 (0.000171122) 	 BC Loss: 0.000019616 (0.000025725)
Iteration: [ 25001/ 50000] 	 Loss: 0.000203542 (0.000227654) 	 Physics Loss: 0.000042980 (0.000038498) 	 Data Loss: 0.000141263 (0.000168537) 	 BC Loss: 0.000019298 (0.000020620)
Iteration: [ 26001/ 50000] 	 Loss: 0.000221975 (0.000241762) 	 Physics Loss: 0.000047288 (0.000053857) 	 Data Loss: 0.000155904 (0.000164544) 	 BC Loss: 0.000018782 (0.000023362)
Iteration: [ 27001/ 50000] 	 Loss: 0.000221044 (0.000227297) 	 Physics Loss: 0.000043223 (0.000041246) 	 Data Loss: 0.000153653 (0.000165061) 	 BC Loss: 0.000024168 (0.000020989)
Iteration: [ 28001/ 50000] 	 Loss: 0.000206697 (0.000216978) 	 Physics Loss: 0.000036950 (0.000038347) 	 Data Loss: 0.000152914 (0.000158207) 	 BC Loss: 0.000016832 (0.000020425)
Iteration: [ 29001/ 50000] 	 Loss: 0.000198937 (0.000227398) 	 Physics Loss: 0.000030450 (0.000048362) 	 Data Loss: 0.000139952 (0.000158875) 	 BC Loss: 0.000028535 (0.000020161)
Iteration: [ 30001/ 50000] 	 Loss: 0.000196092 (0.000214346) 	 Physics Loss: 0.000021757 (0.000034069) 	 Data Loss: 0.000154593 (0.000160857) 	 BC Loss: 0.000019742 (0.000019420)
Iteration: [ 31001/ 50000] 	 Loss: 0.000265356 (0.000219053) 	 Physics Loss: 0.000027731 (0.000040833) 	 Data Loss: 0.000217966 (0.000158773) 	 BC Loss: 0.000019659 (0.000019446)
Iteration: [ 32001/ 50000] 	 Loss: 0.000204912 (0.000201352) 	 Physics Loss: 0.000035430 (0.000030013) 	 Data Loss: 0.000149537 (0.000152606) 	 BC Loss: 0.000019945 (0.000018734)
Iteration: [ 33001/ 50000] 	 Loss: 0.000175845 (0.000196797) 	 Physics Loss: 0.000023642 (0.000029965) 	 Data Loss: 0.000132320 (0.000147994) 	 BC Loss: 0.000019883 (0.000018839)
Iteration: [ 34001/ 50000] 	 Loss: 0.000193787 (0.000198346) 	 Physics Loss: 0.000030744 (0.000033351) 	 Data Loss: 0.000142107 (0.000145769) 	 BC Loss: 0.000020936 (0.000019226)
Iteration: [ 35001/ 50000] 	 Loss: 0.000128827 (0.000203731) 	 Physics Loss: 0.000014563 (0.000038067) 	 Data Loss: 0.000094805 (0.000147789) 	 BC Loss: 0.000019459 (0.000017875)
Iteration: [ 36001/ 50000] 	 Loss: 0.000172073 (0.000200649) 	 Physics Loss: 0.000033216 (0.000035199) 	 Data Loss: 0.000119202 (0.000147519) 	 BC Loss: 0.000019655 (0.000017931)
Iteration: [ 37001/ 50000] 	 Loss: 0.000292938 (0.000190241) 	 Physics Loss: 0.000059189 (0.000026024) 	 Data Loss: 0.000216438 (0.000145679) 	 BC Loss: 0.000017311 (0.000018539)
Iteration: [ 38001/ 50000] 	 Loss: 0.000240104 (0.000195396) 	 Physics Loss: 0.000023660 (0.000027009) 	 Data Loss: 0.000193874 (0.000152148) 	 BC Loss: 0.000022570 (0.000016239)
Iteration: [ 39001/ 50000] 	 Loss: 0.000168815 (0.000193866) 	 Physics Loss: 0.000031025 (0.000029484) 	 Data Loss: 0.000122587 (0.000146424) 	 BC Loss: 0.000015203 (0.000017958)
Iteration: [ 40001/ 50000] 	 Loss: 0.000169602 (0.000191625) 	 Physics Loss: 0.000022515 (0.000028350) 	 Data Loss: 0.000127267 (0.000145208) 	 BC Loss: 0.000019820 (0.000018067)
Iteration: [ 41001/ 50000] 	 Loss: 0.000152231 (0.000187496) 	 Physics Loss: 0.000015423 (0.000023362) 	 Data Loss: 0.000120504 (0.000147267) 	 BC Loss: 0.000016305 (0.000016868)
Iteration: [ 42001/ 50000] 	 Loss: 0.000184242 (0.000191716) 	 Physics Loss: 0.000033204 (0.000027409) 	 Data Loss: 0.000136543 (0.000146515) 	 BC Loss: 0.000014495 (0.000017792)
Iteration: [ 43001/ 50000] 	 Loss: 0.000193355 (0.000195349) 	 Physics Loss: 0.000028922 (0.000028754) 	 Data Loss: 0.000151893 (0.000146339) 	 BC Loss: 0.000012539 (0.000020256)
Iteration: [ 44001/ 50000] 	 Loss: 0.000173565 (0.000190991) 	 Physics Loss: 0.000015205 (0.000026691) 	 Data Loss: 0.000143164 (0.000143627) 	 BC Loss: 0.000015196 (0.000020673)
Iteration: [ 45001/ 50000] 	 Loss: 0.000271715 (0.000211902) 	 Physics Loss: 0.000040143 (0.000037920) 	 Data Loss: 0.000205714 (0.000152970) 	 BC Loss: 0.000025858 (0.000021011)
Iteration: [ 46001/ 50000] 	 Loss: 0.000203876 (0.000183473) 	 Physics Loss: 0.000029680 (0.000027353) 	 Data Loss: 0.000156417 (0.000139164) 	 BC Loss: 0.000017779 (0.000016956)
Iteration: [ 47001/ 50000] 	 Loss: 0.000167380 (0.000183199) 	 Physics Loss: 0.000013768 (0.000026078) 	 Data Loss: 0.000137723 (0.000139940) 	 BC Loss: 0.000015889 (0.000017181)
Iteration: [ 48001/ 50000] 	 Loss: 0.000187514 (0.000180317) 	 Physics Loss: 0.000027903 (0.000024574) 	 Data Loss: 0.000142597 (0.000138718) 	 BC Loss: 0.000017015 (0.000017025)
Iteration: [ 49001/ 50000] 	 Loss: 0.000164184 (0.000193265) 	 Physics Loss: 0.000018983 (0.000033103) 	 Data Loss: 0.000124857 (0.000143166) 	 BC Loss: 0.000020345 (0.000016995)

```

## Visualizing the Results {#Visualizing-the-Results}

```julia
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
```

## Appendix {#Appendix}

```julia
using InteractiveUtils
InteractiveUtils.versioninfo()

if @isdefined(MLDataDevices)
    if @isdefined(CUDA) && MLDataDevices.functional(CUDADevice)
        println()
        CUDA.versioninfo()
    end

    if @isdefined(AMDGPU) && MLDataDevices.functional(AMDGPUDevice)
        println()
        AMDGPU.versioninfo()
    end
end

```

```
Julia Version 1.12.7
Commit 6d172b025e4 (2026-08-15 08:05 UTC)
Build Info:
  Official https://julialang.org release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 4 × AMD EPYC 7763 64-Core Processor
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, znver3)
  GC: Built with stock GC
Threads: 4 default, 1 interactive, 4 GC (on 4 virtual cores)
Environment:
  JULIA_DEBUG = Literate
  LD_LIBRARY_PATH = 
  JULIA_NUM_THREADS = 4
  JULIA_CPU_HARD_MEMORY_LIMIT = 100%
  JULIA_PKG_PRECOMPILE_AUTO = 0

```

***

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
