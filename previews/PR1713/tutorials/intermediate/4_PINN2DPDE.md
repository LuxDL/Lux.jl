---
url: /previews/PR1713/tutorials/intermediate/4_PINN2DPDE.md
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
Iteration: [     1/ 50000] 	 Loss: 20.523933411 (20.523933411) 	 Physics Loss: 16.931318283 (16.931318283) 	 Data Loss: 2.007483006 (2.007483006) 	 BC Loss: 1.585133076 (1.585133076)
Iteration: [  1001/ 50000] 	 Loss: 0.017368633 (0.019241149) 	 Physics Loss: 0.000384369 (0.000523634) 	 Data Loss: 0.005318495 (0.007538573) 	 BC Loss: 0.011665769 (0.011178941)
Iteration: [  2001/ 50000] 	 Loss: 0.015431687 (0.018665662) 	 Physics Loss: 0.001248591 (0.001662040) 	 Data Loss: 0.004322526 (0.006408236) 	 BC Loss: 0.009860570 (0.010595386)
Iteration: [  3001/ 50000] 	 Loss: 0.015749766 (0.015216023) 	 Physics Loss: 0.000569926 (0.001279073) 	 Data Loss: 0.004014893 (0.004232432) 	 BC Loss: 0.011164947 (0.009704518)
Iteration: [  4001/ 50000] 	 Loss: 0.009720733 (0.008718354) 	 Physics Loss: 0.002388266 (0.003384499) 	 Data Loss: 0.003175691 (0.002104893) 	 BC Loss: 0.004156777 (0.003228962)
Iteration: [  5001/ 50000] 	 Loss: 0.005178656 (0.006881488) 	 Physics Loss: 0.002358951 (0.003289018) 	 Data Loss: 0.001529565 (0.001697533) 	 BC Loss: 0.001290140 (0.001894937)
Iteration: [  6001/ 50000] 	 Loss: 0.000997154 (0.001254963) 	 Physics Loss: 0.000242135 (0.000291160) 	 Data Loss: 0.000560088 (0.000739742) 	 BC Loss: 0.000194932 (0.000224061)
Iteration: [  7001/ 50000] 	 Loss: 0.001353245 (0.000882442) 	 Physics Loss: 0.000298322 (0.000256934) 	 Data Loss: 0.000959968 (0.000507651) 	 BC Loss: 0.000094955 (0.000117857)
Iteration: [  8001/ 50000] 	 Loss: 0.000929818 (0.000924655) 	 Physics Loss: 0.000506129 (0.000412100) 	 Data Loss: 0.000322992 (0.000413168) 	 BC Loss: 0.000100697 (0.000099387)
Iteration: [  9001/ 50000] 	 Loss: 0.002412320 (0.002617365) 	 Physics Loss: 0.001345087 (0.001751287) 	 Data Loss: 0.000717711 (0.000492646) 	 BC Loss: 0.000349522 (0.000373431)
Iteration: [ 10001/ 50000] 	 Loss: 0.000664190 (0.000827310) 	 Physics Loss: 0.000264148 (0.000447259) 	 Data Loss: 0.000322410 (0.000314508) 	 BC Loss: 0.000077632 (0.000065543)
Iteration: [ 11001/ 50000] 	 Loss: 0.000398029 (0.000374892) 	 Physics Loss: 0.000172163 (0.000065384) 	 Data Loss: 0.000182478 (0.000273066) 	 BC Loss: 0.000043388 (0.000036442)
Iteration: [ 12001/ 50000] 	 Loss: 0.000265263 (0.000349337) 	 Physics Loss: 0.000051372 (0.000063076) 	 Data Loss: 0.000174540 (0.000249513) 	 BC Loss: 0.000039350 (0.000036749)
Iteration: [ 13001/ 50000] 	 Loss: 0.000311561 (0.000327085) 	 Physics Loss: 0.000069026 (0.000064755) 	 Data Loss: 0.000207745 (0.000228380) 	 BC Loss: 0.000034790 (0.000033950)
Iteration: [ 14001/ 50000] 	 Loss: 0.000391938 (0.000331997) 	 Physics Loss: 0.000064823 (0.000065472) 	 Data Loss: 0.000292252 (0.000237302) 	 BC Loss: 0.000034863 (0.000029223)
Iteration: [ 15001/ 50000] 	 Loss: 0.000255243 (0.000286342) 	 Physics Loss: 0.000056821 (0.000053896) 	 Data Loss: 0.000168645 (0.000201565) 	 BC Loss: 0.000029776 (0.000030882)
Iteration: [ 16001/ 50000] 	 Loss: 0.000262630 (0.000291149) 	 Physics Loss: 0.000083043 (0.000058836) 	 Data Loss: 0.000143263 (0.000203773) 	 BC Loss: 0.000036324 (0.000028539)
Iteration: [ 17001/ 50000] 	 Loss: 0.000443640 (0.000306880) 	 Physics Loss: 0.000101899 (0.000070632) 	 Data Loss: 0.000316872 (0.000205646) 	 BC Loss: 0.000024870 (0.000030602)
Iteration: [ 18001/ 50000] 	 Loss: 0.000214344 (0.000275891) 	 Physics Loss: 0.000044877 (0.000052073) 	 Data Loss: 0.000140145 (0.000197594) 	 BC Loss: 0.000029321 (0.000026224)
Iteration: [ 19001/ 50000] 	 Loss: 0.000215638 (0.000284458) 	 Physics Loss: 0.000061368 (0.000057591) 	 Data Loss: 0.000135524 (0.000201337) 	 BC Loss: 0.000018747 (0.000025530)
Iteration: [ 20001/ 50000] 	 Loss: 0.000300323 (0.000254175) 	 Physics Loss: 0.000042763 (0.000048010) 	 Data Loss: 0.000238605 (0.000182547) 	 BC Loss: 0.000018955 (0.000023618)
Iteration: [ 21001/ 50000] 	 Loss: 0.000285537 (0.000245628) 	 Physics Loss: 0.000033752 (0.000049865) 	 Data Loss: 0.000225020 (0.000172785) 	 BC Loss: 0.000026765 (0.000022977)
Iteration: [ 22001/ 50000] 	 Loss: 0.000171679 (0.000241108) 	 Physics Loss: 0.000035671 (0.000050116) 	 Data Loss: 0.000109144 (0.000169154) 	 BC Loss: 0.000026863 (0.000021838)
Iteration: [ 23001/ 50000] 	 Loss: 0.000220608 (0.000249622) 	 Physics Loss: 0.000031079 (0.000049382) 	 Data Loss: 0.000167648 (0.000178082) 	 BC Loss: 0.000021881 (0.000022157)
Iteration: [ 24001/ 50000] 	 Loss: 0.000289156 (0.000244626) 	 Physics Loss: 0.000045353 (0.000048354) 	 Data Loss: 0.000223994 (0.000171661) 	 BC Loss: 0.000019809 (0.000024611)
Iteration: [ 25001/ 50000] 	 Loss: 0.000196876 (0.000224533) 	 Physics Loss: 0.000033634 (0.000034957) 	 Data Loss: 0.000143806 (0.000168820) 	 BC Loss: 0.000019436 (0.000020756)
Iteration: [ 26001/ 50000] 	 Loss: 0.000242916 (0.000273428) 	 Physics Loss: 0.000063267 (0.000083063) 	 Data Loss: 0.000159937 (0.000165268) 	 BC Loss: 0.000019711 (0.000025096)
Iteration: [ 27001/ 50000] 	 Loss: 0.000220491 (0.000231191) 	 Physics Loss: 0.000043208 (0.000044246) 	 Data Loss: 0.000155551 (0.000165480) 	 BC Loss: 0.000021732 (0.000021464)
Iteration: [ 28001/ 50000] 	 Loss: 0.000209593 (0.000224092) 	 Physics Loss: 0.000040483 (0.000044365) 	 Data Loss: 0.000151889 (0.000158626) 	 BC Loss: 0.000017221 (0.000021100)
Iteration: [ 29001/ 50000] 	 Loss: 0.000221173 (0.000226324) 	 Physics Loss: 0.000055902 (0.000047936) 	 Data Loss: 0.000137886 (0.000158384) 	 BC Loss: 0.000027385 (0.000020004)
Iteration: [ 30001/ 50000] 	 Loss: 0.000216761 (0.000234396) 	 Physics Loss: 0.000036500 (0.000051070) 	 Data Loss: 0.000157228 (0.000162147) 	 BC Loss: 0.000023033 (0.000021179)
Iteration: [ 31001/ 50000] 	 Loss: 0.000279395 (0.000244237) 	 Physics Loss: 0.000044799 (0.000062332) 	 Data Loss: 0.000213954 (0.000160094) 	 BC Loss: 0.000020642 (0.000021811)
Iteration: [ 32001/ 50000] 	 Loss: 0.000213101 (0.000202114) 	 Physics Loss: 0.000044648 (0.000030874) 	 Data Loss: 0.000148684 (0.000152272) 	 BC Loss: 0.000019769 (0.000018968)
Iteration: [ 33001/ 50000] 	 Loss: 0.000176304 (0.000195908) 	 Physics Loss: 0.000024516 (0.000029386) 	 Data Loss: 0.000130967 (0.000147371) 	 BC Loss: 0.000020822 (0.000019151)
Iteration: [ 34001/ 50000] 	 Loss: 0.000196155 (0.000199011) 	 Physics Loss: 0.000033418 (0.000035223) 	 Data Loss: 0.000143324 (0.000144958) 	 BC Loss: 0.000019413 (0.000018831)
Iteration: [ 35001/ 50000] 	 Loss: 0.000134873 (0.000210021) 	 Physics Loss: 0.000021186 (0.000044488) 	 Data Loss: 0.000093939 (0.000147142) 	 BC Loss: 0.000019748 (0.000018391)
Iteration: [ 36001/ 50000] 	 Loss: 0.000169097 (0.000196988) 	 Physics Loss: 0.000032668 (0.000031148) 	 Data Loss: 0.000116275 (0.000147139) 	 BC Loss: 0.000020154 (0.000018701)
Iteration: [ 37001/ 50000] 	 Loss: 0.000297034 (0.000189876) 	 Physics Loss: 0.000060284 (0.000025958) 	 Data Loss: 0.000218353 (0.000145206) 	 BC Loss: 0.000018397 (0.000018713)
Iteration: [ 38001/ 50000] 	 Loss: 0.000244587 (0.000212520) 	 Physics Loss: 0.000032065 (0.000042292) 	 Data Loss: 0.000190624 (0.000152526) 	 BC Loss: 0.000021897 (0.000017701)
Iteration: [ 39001/ 50000] 	 Loss: 0.000166397 (0.000192370) 	 Physics Loss: 0.000026274 (0.000028614) 	 Data Loss: 0.000126094 (0.000145622) 	 BC Loss: 0.000014029 (0.000018134)
Iteration: [ 40001/ 50000] 	 Loss: 0.000171062 (0.000190409) 	 Physics Loss: 0.000026979 (0.000027756) 	 Data Loss: 0.000123709 (0.000143877) 	 BC Loss: 0.000020374 (0.000018776)
Iteration: [ 41001/ 50000] 	 Loss: 0.000156403 (0.000197690) 	 Physics Loss: 0.000019827 (0.000033383) 	 Data Loss: 0.000119207 (0.000146459) 	 BC Loss: 0.000017369 (0.000017848)
Iteration: [ 42001/ 50000] 	 Loss: 0.000175119 (0.000192770) 	 Physics Loss: 0.000023378 (0.000029362) 	 Data Loss: 0.000135567 (0.000145001) 	 BC Loss: 0.000016174 (0.000018407)
Iteration: [ 43001/ 50000] 	 Loss: 0.000195367 (0.000198087) 	 Physics Loss: 0.000031742 (0.000034099) 	 Data Loss: 0.000148359 (0.000144140) 	 BC Loss: 0.000015267 (0.000019847)
Iteration: [ 44001/ 50000] 	 Loss: 0.000173652 (0.000191548) 	 Physics Loss: 0.000017230 (0.000028143) 	 Data Loss: 0.000143066 (0.000142565) 	 BC Loss: 0.000013356 (0.000020840)
Iteration: [ 45001/ 50000] 	 Loss: 0.000271109 (0.000208844) 	 Physics Loss: 0.000041909 (0.000036187) 	 Data Loss: 0.000205018 (0.000151504) 	 BC Loss: 0.000024181 (0.000021152)
Iteration: [ 46001/ 50000] 	 Loss: 0.000195127 (0.000179302) 	 Physics Loss: 0.000021549 (0.000023813) 	 Data Loss: 0.000156260 (0.000138296) 	 BC Loss: 0.000017317 (0.000017192)
Iteration: [ 47001/ 50000] 	 Loss: 0.000180578 (0.000183945) 	 Physics Loss: 0.000027255 (0.000027695) 	 Data Loss: 0.000135071 (0.000138914) 	 BC Loss: 0.000018252 (0.000017336)
Iteration: [ 48001/ 50000] 	 Loss: 0.000180244 (0.000183338) 	 Physics Loss: 0.000019625 (0.000027794) 	 Data Loss: 0.000141994 (0.000137860) 	 BC Loss: 0.000018625 (0.000017685)
Iteration: [ 49001/ 50000] 	 Loss: 0.000170717 (0.000193546) 	 Physics Loss: 0.000028078 (0.000035036) 	 Data Loss: 0.000122452 (0.000140633) 	 BC Loss: 0.000020187 (0.000017877)

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
Julia Version 1.12.6
Commit 15346901f00 (2026-04-09 19:20 UTC)
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
