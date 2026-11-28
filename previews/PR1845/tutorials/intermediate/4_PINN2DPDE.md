---
url: /previews/PR1845/tutorials/intermediate/4_PINN2DPDE.md
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
Iteration: [  1001/ 50000] 	 Loss: 0.017368639 (0.019241150) 	 Physics Loss: 0.000384361 (0.000523638) 	 Data Loss: 0.005318498 (0.007538575) 	 BC Loss: 0.011665780 (0.011178941)
Iteration: [  2001/ 50000] 	 Loss: 0.015431677 (0.018665684) 	 Physics Loss: 0.001248581 (0.001662052) 	 Data Loss: 0.004322527 (0.006408239) 	 BC Loss: 0.009860569 (0.010595391)
Iteration: [  3001/ 50000] 	 Loss: 0.015749754 (0.015216049) 	 Physics Loss: 0.000569920 (0.001279087) 	 Data Loss: 0.004014898 (0.004232440) 	 BC Loss: 0.011164936 (0.009704523)
Iteration: [  4001/ 50000] 	 Loss: 0.009719364 (0.008715075) 	 Physics Loss: 0.002387916 (0.003381869) 	 Data Loss: 0.003175879 (0.002104684) 	 BC Loss: 0.004155569 (0.003228523)
Iteration: [  5001/ 50000] 	 Loss: 0.004781279 (0.005924764) 	 Physics Loss: 0.001635808 (0.002552078) 	 Data Loss: 0.001893939 (0.001704932) 	 BC Loss: 0.001251532 (0.001667753)
Iteration: [  6001/ 50000] 	 Loss: 0.001141098 (0.001293444) 	 Physics Loss: 0.000369332 (0.000327948) 	 Data Loss: 0.000562205 (0.000711534) 	 BC Loss: 0.000209562 (0.000253962)
Iteration: [  7001/ 50000] 	 Loss: 0.001373543 (0.000914103) 	 Physics Loss: 0.000339031 (0.000279002) 	 Data Loss: 0.000926378 (0.000499787) 	 BC Loss: 0.000108134 (0.000135314)
Iteration: [  8001/ 50000] 	 Loss: 0.001921344 (0.000914853) 	 Physics Loss: 0.001506431 (0.000418977) 	 Data Loss: 0.000327120 (0.000392794) 	 BC Loss: 0.000087794 (0.000103083)
Iteration: [  9001/ 50000] 	 Loss: 0.001158656 (0.001232725) 	 Physics Loss: 0.000469914 (0.000741426) 	 Data Loss: 0.000574190 (0.000364693) 	 BC Loss: 0.000114553 (0.000126606)
Iteration: [ 10001/ 50000] 	 Loss: 0.001090891 (0.000955040) 	 Physics Loss: 0.000681702 (0.000583719) 	 Data Loss: 0.000328558 (0.000299223) 	 BC Loss: 0.000080631 (0.000072099)
Iteration: [ 11001/ 50000] 	 Loss: 0.000426867 (0.000373272) 	 Physics Loss: 0.000197589 (0.000069722) 	 Data Loss: 0.000181862 (0.000265058) 	 BC Loss: 0.000047415 (0.000038492)
Iteration: [ 12001/ 50000] 	 Loss: 0.000264152 (0.000350179) 	 Physics Loss: 0.000052475 (0.000069041) 	 Data Loss: 0.000166885 (0.000243956) 	 BC Loss: 0.000044793 (0.000037182)
Iteration: [ 13001/ 50000] 	 Loss: 0.000300824 (0.000328638) 	 Physics Loss: 0.000061321 (0.000069267) 	 Data Loss: 0.000205423 (0.000223316) 	 BC Loss: 0.000034080 (0.000036055)
Iteration: [ 14001/ 50000] 	 Loss: 0.000379914 (0.000328483) 	 Physics Loss: 0.000067674 (0.000066689) 	 Data Loss: 0.000279995 (0.000231210) 	 BC Loss: 0.000032244 (0.000030583)
Iteration: [ 15001/ 50000] 	 Loss: 0.000241065 (0.000289811) 	 Physics Loss: 0.000042137 (0.000057214) 	 Data Loss: 0.000166851 (0.000199433) 	 BC Loss: 0.000032077 (0.000033164)
Iteration: [ 16001/ 50000] 	 Loss: 0.000239965 (0.000292205) 	 Physics Loss: 0.000060112 (0.000062307) 	 Data Loss: 0.000144785 (0.000200296) 	 BC Loss: 0.000035067 (0.000029601)
Iteration: [ 17001/ 50000] 	 Loss: 0.000410343 (0.000291026) 	 Physics Loss: 0.000084026 (0.000061117) 	 Data Loss: 0.000301545 (0.000200710) 	 BC Loss: 0.000024773 (0.000029199)
Iteration: [ 18001/ 50000] 	 Loss: 0.000216335 (0.000282068) 	 Physics Loss: 0.000037852 (0.000057351) 	 Data Loss: 0.000143773 (0.000195806) 	 BC Loss: 0.000034709 (0.000028912)
Iteration: [ 19001/ 50000] 	 Loss: 0.000210039 (0.000281958) 	 Physics Loss: 0.000051615 (0.000059717) 	 Data Loss: 0.000139723 (0.000196380) 	 BC Loss: 0.000018701 (0.000025861)
Iteration: [ 20001/ 50000] 	 Loss: 0.000309365 (0.000248267) 	 Physics Loss: 0.000065262 (0.000045076) 	 Data Loss: 0.000223451 (0.000180540) 	 BC Loss: 0.000020652 (0.000022651)
Iteration: [ 21001/ 50000] 	 Loss: 0.000290638 (0.000242398) 	 Physics Loss: 0.000044906 (0.000047455) 	 Data Loss: 0.000220310 (0.000171069) 	 BC Loss: 0.000025422 (0.000023875)
Iteration: [ 22001/ 50000] 	 Loss: 0.000161331 (0.000237040) 	 Physics Loss: 0.000025615 (0.000046042) 	 Data Loss: 0.000108555 (0.000167683) 	 BC Loss: 0.000027161 (0.000023314)
Iteration: [ 23001/ 50000] 	 Loss: 0.000236465 (0.000253260) 	 Physics Loss: 0.000045185 (0.000053715) 	 Data Loss: 0.000167657 (0.000177433) 	 BC Loss: 0.000023623 (0.000022112)
Iteration: [ 24001/ 50000] 	 Loss: 0.000280694 (0.000245937) 	 Physics Loss: 0.000044759 (0.000051174) 	 Data Loss: 0.000216626 (0.000169969) 	 BC Loss: 0.000019309 (0.000024794)
Iteration: [ 25001/ 50000] 	 Loss: 0.000205393 (0.000228544) 	 Physics Loss: 0.000043707 (0.000039053) 	 Data Loss: 0.000143174 (0.000167665) 	 BC Loss: 0.000018512 (0.000021826)
Iteration: [ 26001/ 50000] 	 Loss: 0.000226201 (0.000239777) 	 Physics Loss: 0.000049011 (0.000052645) 	 Data Loss: 0.000155669 (0.000164653) 	 BC Loss: 0.000021521 (0.000022479)
Iteration: [ 27001/ 50000] 	 Loss: 0.000234833 (0.000242097) 	 Physics Loss: 0.000054915 (0.000053867) 	 Data Loss: 0.000154327 (0.000165027) 	 BC Loss: 0.000025590 (0.000023203)
Iteration: [ 28001/ 50000] 	 Loss: 0.000217161 (0.000216690) 	 Physics Loss: 0.000044683 (0.000036386) 	 Data Loss: 0.000153870 (0.000158530) 	 BC Loss: 0.000018608 (0.000021775)
Iteration: [ 29001/ 50000] 	 Loss: 0.000208464 (0.000233676) 	 Physics Loss: 0.000030726 (0.000052708) 	 Data Loss: 0.000146229 (0.000159568) 	 BC Loss: 0.000031510 (0.000021400)
Iteration: [ 30001/ 50000] 	 Loss: 0.000214761 (0.000226477) 	 Physics Loss: 0.000029807 (0.000043603) 	 Data Loss: 0.000157193 (0.000161268) 	 BC Loss: 0.000027761 (0.000021607)
Iteration: [ 31001/ 50000] 	 Loss: 0.000265199 (0.000217124) 	 Physics Loss: 0.000036269 (0.000037661) 	 Data Loss: 0.000208184 (0.000158302) 	 BC Loss: 0.000020745 (0.000021161)
Iteration: [ 32001/ 50000] 	 Loss: 0.000210835 (0.000215451) 	 Physics Loss: 0.000045787 (0.000042326) 	 Data Loss: 0.000145907 (0.000152651) 	 BC Loss: 0.000019141 (0.000020473)
Iteration: [ 33001/ 50000] 	 Loss: 0.000191795 (0.000207433) 	 Physics Loss: 0.000033309 (0.000037275) 	 Data Loss: 0.000137259 (0.000149430) 	 BC Loss: 0.000021227 (0.000020728)
Iteration: [ 34001/ 50000] 	 Loss: 0.000186796 (0.000198027) 	 Physics Loss: 0.000025182 (0.000032559) 	 Data Loss: 0.000141554 (0.000145818) 	 BC Loss: 0.000020060 (0.000019650)
Iteration: [ 35001/ 50000] 	 Loss: 0.000144340 (0.000224535) 	 Physics Loss: 0.000023654 (0.000056309) 	 Data Loss: 0.000097778 (0.000148069) 	 BC Loss: 0.000022908 (0.000020157)
Iteration: [ 36001/ 50000] 	 Loss: 0.000165921 (0.000199502) 	 Physics Loss: 0.000033062 (0.000032932) 	 Data Loss: 0.000113460 (0.000147295) 	 BC Loss: 0.000019399 (0.000019276)
Iteration: [ 37001/ 50000] 	 Loss: 0.000297117 (0.000192969) 	 Physics Loss: 0.000055847 (0.000027335) 	 Data Loss: 0.000222455 (0.000145981) 	 BC Loss: 0.000018815 (0.000019653)
Iteration: [ 38001/ 50000] 	 Loss: 0.000240158 (0.000205162) 	 Physics Loss: 0.000026932 (0.000035226) 	 Data Loss: 0.000188795 (0.000151827) 	 BC Loss: 0.000024431 (0.000018109)
Iteration: [ 39001/ 50000] 	 Loss: 0.000177084 (0.000205383) 	 Physics Loss: 0.000036391 (0.000039636) 	 Data Loss: 0.000124248 (0.000146513) 	 BC Loss: 0.000016445 (0.000019233)
Iteration: [ 40001/ 50000] 	 Loss: 0.000170448 (0.000191818) 	 Physics Loss: 0.000022863 (0.000027464) 	 Data Loss: 0.000126438 (0.000144768) 	 BC Loss: 0.000021147 (0.000019586)
Iteration: [ 41001/ 50000] 	 Loss: 0.000168595 (0.000204042) 	 Physics Loss: 0.000027855 (0.000036996) 	 Data Loss: 0.000120801 (0.000147679) 	 BC Loss: 0.000019939 (0.000019367)
Iteration: [ 42001/ 50000] 	 Loss: 0.000178793 (0.000192347) 	 Physics Loss: 0.000026237 (0.000027532) 	 Data Loss: 0.000137333 (0.000145837) 	 BC Loss: 0.000015223 (0.000018979)
Iteration: [ 43001/ 50000] 	 Loss: 0.000197070 (0.000196369) 	 Physics Loss: 0.000028221 (0.000030472) 	 Data Loss: 0.000152970 (0.000145309) 	 BC Loss: 0.000015880 (0.000020588)
Iteration: [ 44001/ 50000] 	 Loss: 0.000176421 (0.000191119) 	 Physics Loss: 0.000015855 (0.000026995) 	 Data Loss: 0.000146114 (0.000143455) 	 BC Loss: 0.000014451 (0.000020669)
Iteration: [ 45001/ 50000] 	 Loss: 0.000273140 (0.000208721) 	 Physics Loss: 0.000042462 (0.000034932) 	 Data Loss: 0.000203292 (0.000152183) 	 BC Loss: 0.000027386 (0.000021606)
Iteration: [ 46001/ 50000] 	 Loss: 0.000215800 (0.000198168) 	 Physics Loss: 0.000035776 (0.000040078) 	 Data Loss: 0.000159717 (0.000139368) 	 BC Loss: 0.000020307 (0.000018722)
Iteration: [ 47001/ 50000] 	 Loss: 0.000187659 (0.000190407) 	 Physics Loss: 0.000033325 (0.000031132) 	 Data Loss: 0.000137487 (0.000140385) 	 BC Loss: 0.000016848 (0.000018891)
Iteration: [ 48001/ 50000] 	 Loss: 0.000196416 (0.000179751) 	 Physics Loss: 0.000034803 (0.000022792) 	 Data Loss: 0.000143197 (0.000138272) 	 BC Loss: 0.000018416 (0.000018687)
Iteration: [ 49001/ 50000] 	 Loss: 0.000174183 (0.000207697) 	 Physics Loss: 0.000029163 (0.000047410) 	 Data Loss: 0.000122030 (0.000141347) 	 BC Loss: 0.000022990 (0.000018939)

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
  CPU: 4 × Intel(R) Xeon(R) 6975P-C
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, graniterapids)
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
