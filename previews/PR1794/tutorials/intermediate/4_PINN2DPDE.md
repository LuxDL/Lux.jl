---
url: /previews/PR1794/tutorials/intermediate/4_PINN2DPDE.md
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
Iteration: [     1/ 50000] 	 Loss: 20.523931503 (20.523931503) 	 Physics Loss: 16.931316376 (16.931316376) 	 Data Loss: 2.007482767 (2.007482767) 	 BC Loss: 1.585133195 (1.585133195)
Iteration: [  1001/ 50000] 	 Loss: 0.017368648 (0.019241160) 	 Physics Loss: 0.000384364 (0.000523640) 	 Data Loss: 0.005318501 (0.007538575) 	 BC Loss: 0.011665783 (0.011178943)
Iteration: [  2001/ 50000] 	 Loss: 0.015431638 (0.018665679) 	 Physics Loss: 0.001248533 (0.001662055) 	 Data Loss: 0.004322531 (0.006408238) 	 BC Loss: 0.009860574 (0.010595386)
Iteration: [  3001/ 50000] 	 Loss: 0.015749730 (0.015216024) 	 Physics Loss: 0.000569911 (0.001279077) 	 Data Loss: 0.004014886 (0.004232429) 	 BC Loss: 0.011164933 (0.009704515)
Iteration: [  4001/ 50000] 	 Loss: 0.009720150 (0.008717315) 	 Physics Loss: 0.002388107 (0.003383615) 	 Data Loss: 0.003175774 (0.002104835) 	 BC Loss: 0.004156270 (0.003228865)
Iteration: [  5001/ 50000] 	 Loss: 0.008058457 (0.006213902) 	 Physics Loss: 0.004704554 (0.003081223) 	 Data Loss: 0.001870022 (0.001497522) 	 BC Loss: 0.001483880 (0.001635158)
Iteration: [  6001/ 50000] 	 Loss: 0.001036036 (0.001236259) 	 Physics Loss: 0.000285777 (0.000285752) 	 Data Loss: 0.000569707 (0.000734372) 	 BC Loss: 0.000180552 (0.000216135)
Iteration: [  7001/ 50000] 	 Loss: 0.001357321 (0.000937879) 	 Physics Loss: 0.000303574 (0.000310826) 	 Data Loss: 0.000954314 (0.000503178) 	 BC Loss: 0.000099433 (0.000123875)
Iteration: [  8001/ 50000] 	 Loss: 0.001480718 (0.001077451) 	 Physics Loss: 0.001034799 (0.000529366) 	 Data Loss: 0.000333786 (0.000429206) 	 BC Loss: 0.000112133 (0.000118878)
Iteration: [  9001/ 50000] 	 Loss: 0.001823213 (0.002750510) 	 Physics Loss: 0.000910164 (0.001843044) 	 Data Loss: 0.000737052 (0.000550730) 	 BC Loss: 0.000175997 (0.000356737)
Iteration: [ 10001/ 50000] 	 Loss: 0.000588715 (0.000639002) 	 Physics Loss: 0.000230028 (0.000280006) 	 Data Loss: 0.000296212 (0.000301793) 	 BC Loss: 0.000062475 (0.000057203)
Iteration: [ 11001/ 50000] 	 Loss: 0.000376688 (0.000381517) 	 Physics Loss: 0.000147406 (0.000070519) 	 Data Loss: 0.000185168 (0.000274042) 	 BC Loss: 0.000044114 (0.000036956)
Iteration: [ 12001/ 50000] 	 Loss: 0.000271988 (0.000356739) 	 Physics Loss: 0.000058426 (0.000068989) 	 Data Loss: 0.000170954 (0.000251359) 	 BC Loss: 0.000042608 (0.000036391)
Iteration: [ 13001/ 50000] 	 Loss: 0.000304607 (0.000330138) 	 Physics Loss: 0.000061695 (0.000067998) 	 Data Loss: 0.000207711 (0.000227920) 	 BC Loss: 0.000035201 (0.000034219)
Iteration: [ 14001/ 50000] 	 Loss: 0.000387286 (0.000336998) 	 Physics Loss: 0.000067533 (0.000071328) 	 Data Loss: 0.000286022 (0.000236444) 	 BC Loss: 0.000033731 (0.000029225)
Iteration: [ 15001/ 50000] 	 Loss: 0.000244525 (0.000294403) 	 Physics Loss: 0.000046646 (0.000061055) 	 Data Loss: 0.000167057 (0.000201284) 	 BC Loss: 0.000030822 (0.000032064)
Iteration: [ 16001/ 50000] 	 Loss: 0.000219182 (0.000292848) 	 Physics Loss: 0.000044970 (0.000060844) 	 Data Loss: 0.000143328 (0.000202627) 	 BC Loss: 0.000030884 (0.000029377)
Iteration: [ 17001/ 50000] 	 Loss: 0.000418284 (0.000300559) 	 Physics Loss: 0.000088901 (0.000068167) 	 Data Loss: 0.000305145 (0.000203798) 	 BC Loss: 0.000024237 (0.000028594)
Iteration: [ 18001/ 50000] 	 Loss: 0.000214540 (0.000283577) 	 Physics Loss: 0.000041145 (0.000059568) 	 Data Loss: 0.000139459 (0.000196051) 	 BC Loss: 0.000033936 (0.000027958)
Iteration: [ 19001/ 50000] 	 Loss: 0.000209697 (0.000276129) 	 Physics Loss: 0.000053733 (0.000053294) 	 Data Loss: 0.000136543 (0.000197564) 	 BC Loss: 0.000019422 (0.000025271)
Iteration: [ 20001/ 50000] 	 Loss: 0.000297437 (0.000250475) 	 Physics Loss: 0.000048597 (0.000046034) 	 Data Loss: 0.000228551 (0.000181296) 	 BC Loss: 0.000020289 (0.000023144)
Iteration: [ 21001/ 50000] 	 Loss: 0.000315586 (0.000249625) 	 Physics Loss: 0.000068387 (0.000054444) 	 Data Loss: 0.000224224 (0.000171269) 	 BC Loss: 0.000022975 (0.000023912)
Iteration: [ 22001/ 50000] 	 Loss: 0.000163328 (0.000242072) 	 Physics Loss: 0.000028527 (0.000050208) 	 Data Loss: 0.000109804 (0.000168490) 	 BC Loss: 0.000024998 (0.000023374)
Iteration: [ 23001/ 50000] 	 Loss: 0.000227369 (0.000250272) 	 Physics Loss: 0.000033971 (0.000050191) 	 Data Loss: 0.000169514 (0.000177629) 	 BC Loss: 0.000023884 (0.000022452)
Iteration: [ 24001/ 50000] 	 Loss: 0.000282227 (0.000249104) 	 Physics Loss: 0.000045170 (0.000052736) 	 Data Loss: 0.000217033 (0.000170744) 	 BC Loss: 0.000020023 (0.000025625)
Iteration: [ 25001/ 50000] 	 Loss: 0.000204666 (0.000224447) 	 Physics Loss: 0.000042490 (0.000034780) 	 Data Loss: 0.000143430 (0.000168484) 	 BC Loss: 0.000018746 (0.000021183)
Iteration: [ 26001/ 50000] 	 Loss: 0.000225609 (0.000239271) 	 Physics Loss: 0.000048635 (0.000050938) 	 Data Loss: 0.000154901 (0.000164661) 	 BC Loss: 0.000022073 (0.000023672)
Iteration: [ 27001/ 50000] 	 Loss: 0.000226813 (0.000239047) 	 Physics Loss: 0.000046906 (0.000051103) 	 Data Loss: 0.000153156 (0.000165366) 	 BC Loss: 0.000026751 (0.000022578)
Iteration: [ 28001/ 50000] 	 Loss: 0.000233121 (0.000217296) 	 Physics Loss: 0.000060588 (0.000037568) 	 Data Loss: 0.000154344 (0.000158389) 	 BC Loss: 0.000018190 (0.000021339)
Iteration: [ 29001/ 50000] 	 Loss: 0.000200068 (0.000233081) 	 Physics Loss: 0.000027584 (0.000052539) 	 Data Loss: 0.000142560 (0.000159294) 	 BC Loss: 0.000029924 (0.000021248)
Iteration: [ 30001/ 50000] 	 Loss: 0.000200968 (0.000215778) 	 Physics Loss: 0.000021776 (0.000034416) 	 Data Loss: 0.000155707 (0.000160990) 	 BC Loss: 0.000023485 (0.000020372)
Iteration: [ 31001/ 50000] 	 Loss: 0.000269677 (0.000222227) 	 Physics Loss: 0.000033525 (0.000042767) 	 Data Loss: 0.000216307 (0.000158836) 	 BC Loss: 0.000019845 (0.000020624)
Iteration: [ 32001/ 50000] 	 Loss: 0.000204881 (0.000207156) 	 Physics Loss: 0.000036613 (0.000034770) 	 Data Loss: 0.000147926 (0.000152532) 	 BC Loss: 0.000020342 (0.000019854)
Iteration: [ 33001/ 50000] 	 Loss: 0.000178135 (0.000198575) 	 Physics Loss: 0.000023187 (0.000030287) 	 Data Loss: 0.000133544 (0.000148444) 	 BC Loss: 0.000021403 (0.000019843)
Iteration: [ 34001/ 50000] 	 Loss: 0.000197343 (0.000201501) 	 Physics Loss: 0.000035816 (0.000036262) 	 Data Loss: 0.000141828 (0.000145401) 	 BC Loss: 0.000019699 (0.000019837)
Iteration: [ 35001/ 50000] 	 Loss: 0.000135360 (0.000208391) 	 Physics Loss: 0.000017855 (0.000042153) 	 Data Loss: 0.000097286 (0.000147470) 	 BC Loss: 0.000020219 (0.000018768)
Iteration: [ 36001/ 50000] 	 Loss: 0.000163177 (0.000203016) 	 Physics Loss: 0.000029148 (0.000037855) 	 Data Loss: 0.000115316 (0.000146835) 	 BC Loss: 0.000018713 (0.000018326)
Iteration: [ 37001/ 50000] 	 Loss: 0.000315335 (0.000191347) 	 Physics Loss: 0.000077656 (0.000027144) 	 Data Loss: 0.000219823 (0.000145121) 	 BC Loss: 0.000017855 (0.000019082)
Iteration: [ 38001/ 50000] 	 Loss: 0.000242360 (0.000201289) 	 Physics Loss: 0.000028189 (0.000033312) 	 Data Loss: 0.000190898 (0.000151057) 	 BC Loss: 0.000023274 (0.000016920)
Iteration: [ 39001/ 50000] 	 Loss: 0.000167723 (0.000196516) 	 Physics Loss: 0.000028001 (0.000032192) 	 Data Loss: 0.000123344 (0.000145775) 	 BC Loss: 0.000016378 (0.000018549)
Iteration: [ 40001/ 50000] 	 Loss: 0.000171884 (0.000193054) 	 Physics Loss: 0.000027801 (0.000029782) 	 Data Loss: 0.000123769 (0.000144468) 	 BC Loss: 0.000020314 (0.000018804)
Iteration: [ 41001/ 50000] 	 Loss: 0.000155105 (0.000193688) 	 Physics Loss: 0.000018033 (0.000029175) 	 Data Loss: 0.000118668 (0.000146672) 	 BC Loss: 0.000018404 (0.000017841)
Iteration: [ 42001/ 50000] 	 Loss: 0.000181154 (0.000191585) 	 Physics Loss: 0.000031182 (0.000027259) 	 Data Loss: 0.000134723 (0.000145716) 	 BC Loss: 0.000015250 (0.000018611)
Iteration: [ 43001/ 50000] 	 Loss: 0.000203003 (0.000196137) 	 Physics Loss: 0.000036693 (0.000030643) 	 Data Loss: 0.000152269 (0.000145104) 	 BC Loss: 0.000014041 (0.000020390)
Iteration: [ 44001/ 50000] 	 Loss: 0.000169079 (0.000193837) 	 Physics Loss: 0.000012205 (0.000030080) 	 Data Loss: 0.000141782 (0.000142767) 	 BC Loss: 0.000015092 (0.000020991)
Iteration: [ 45001/ 50000] 	 Loss: 0.000272709 (0.000215217) 	 Physics Loss: 0.000043181 (0.000041304) 	 Data Loss: 0.000203111 (0.000152103) 	 BC Loss: 0.000026417 (0.000021809)
Iteration: [ 46001/ 50000] 	 Loss: 0.000216304 (0.000192701) 	 Physics Loss: 0.000039779 (0.000036757) 	 Data Loss: 0.000158486 (0.000138211) 	 BC Loss: 0.000018038 (0.000017733)
Iteration: [ 47001/ 50000] 	 Loss: 0.000171726 (0.000187417) 	 Physics Loss: 0.000017258 (0.000030401) 	 Data Loss: 0.000137657 (0.000139134) 	 BC Loss: 0.000016812 (0.000017882)
Iteration: [ 48001/ 50000] 	 Loss: 0.000190995 (0.000182310) 	 Physics Loss: 0.000030270 (0.000026420) 	 Data Loss: 0.000143018 (0.000138051) 	 BC Loss: 0.000017707 (0.000017839)
Iteration: [ 49001/ 50000] 	 Loss: 0.000177845 (0.000192422) 	 Physics Loss: 0.000035030 (0.000032635) 	 Data Loss: 0.000123371 (0.000142020) 	 BC Loss: 0.000019443 (0.000017768)

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
  CPU: 4 × AMD EPYC 9V45 96-Core Processor
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, znver5)
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
