---
url: /v1/tutorials/intermediate/4_PINN2DPDE.md
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
Iteration: [     1/ 50000] 	 Loss: 19.575366974 (19.575366974) 	 Physics Loss: 15.844472885 (15.844472885) 	 Data Loss: 2.297609568 (2.297609568) 	 BC Loss: 1.433284163 (1.433284163)
Iteration: [  1001/ 50000] 	 Loss: 0.025148194 (0.019106060) 	 Physics Loss: 0.003428019 (0.000862853) 	 Data Loss: 0.009048486 (0.007053737) 	 BC Loss: 0.012671690 (0.011189475)
Iteration: [  2001/ 50000] 	 Loss: 0.021117344 (0.017542778) 	 Physics Loss: 0.000905034 (0.001242532) 	 Data Loss: 0.008763557 (0.005630509) 	 BC Loss: 0.011448753 (0.010669737)
Iteration: [  3001/ 50000] 	 Loss: 0.008555548 (0.010197213) 	 Physics Loss: 0.001972680 (0.001689518) 	 Data Loss: 0.002368507 (0.002719698) 	 BC Loss: 0.004214362 (0.005787997)
Iteration: [  4001/ 50000] 	 Loss: 0.005804617 (0.007514838) 	 Physics Loss: 0.001707781 (0.003110378) 	 Data Loss: 0.001611021 (0.001868265) 	 BC Loss: 0.002485815 (0.002536195)
Iteration: [  5001/ 50000] 	 Loss: 0.016722595 (0.005241366) 	 Physics Loss: 0.014749043 (0.002982603) 	 Data Loss: 0.001386586 (0.001134192) 	 BC Loss: 0.000586966 (0.001124569)
Iteration: [  6001/ 50000] 	 Loss: 0.000750101 (0.000928570) 	 Physics Loss: 0.000203651 (0.000249187) 	 Data Loss: 0.000435011 (0.000562155) 	 BC Loss: 0.000111438 (0.000117228)
Iteration: [  7001/ 50000] 	 Loss: 0.000665710 (0.000797890) 	 Physics Loss: 0.000276990 (0.000293042) 	 Data Loss: 0.000266622 (0.000423056) 	 BC Loss: 0.000122098 (0.000081793)
Iteration: [  8001/ 50000] 	 Loss: 0.000438303 (0.000666611) 	 Physics Loss: 0.000168106 (0.000266065) 	 Data Loss: 0.000204976 (0.000337135) 	 BC Loss: 0.000065221 (0.000063412)
Iteration: [  9001/ 50000] 	 Loss: 0.000316396 (0.000496952) 	 Physics Loss: 0.000121167 (0.000134641) 	 Data Loss: 0.000167129 (0.000309298) 	 BC Loss: 0.000028100 (0.000053013)
Iteration: [ 10001/ 50000] 	 Loss: 0.000590600 (0.000546281) 	 Physics Loss: 0.000279791 (0.000213445) 	 Data Loss: 0.000238049 (0.000276344) 	 BC Loss: 0.000072760 (0.000056492)
Iteration: [ 11001/ 50000] 	 Loss: 0.000393013 (0.000325269) 	 Physics Loss: 0.000060144 (0.000049363) 	 Data Loss: 0.000303820 (0.000239501) 	 BC Loss: 0.000029048 (0.000036406)
Iteration: [ 12001/ 50000] 	 Loss: 0.000256057 (0.000333205) 	 Physics Loss: 0.000041332 (0.000055173) 	 Data Loss: 0.000174388 (0.000241483) 	 BC Loss: 0.000040337 (0.000036550)
Iteration: [ 13001/ 50000] 	 Loss: 0.000246880 (0.000317153) 	 Physics Loss: 0.000041851 (0.000051822) 	 Data Loss: 0.000176638 (0.000229143) 	 BC Loss: 0.000028391 (0.000036188)
Iteration: [ 14001/ 50000] 	 Loss: 0.000260411 (0.000326004) 	 Physics Loss: 0.000041269 (0.000055387) 	 Data Loss: 0.000185128 (0.000237006) 	 BC Loss: 0.000034015 (0.000033611)
Iteration: [ 15001/ 50000] 	 Loss: 0.000226656 (0.000322472) 	 Physics Loss: 0.000037104 (0.000063788) 	 Data Loss: 0.000157643 (0.000226819) 	 BC Loss: 0.000031908 (0.000031865)
Iteration: [ 16001/ 50000] 	 Loss: 0.000304762 (0.000288509) 	 Physics Loss: 0.000053057 (0.000050601) 	 Data Loss: 0.000221525 (0.000206236) 	 BC Loss: 0.000030181 (0.000031672)
Iteration: [ 17001/ 50000] 	 Loss: 0.000379137 (0.000279876) 	 Physics Loss: 0.000062173 (0.000047327) 	 Data Loss: 0.000267340 (0.000203691) 	 BC Loss: 0.000049623 (0.000028858)
Iteration: [ 18001/ 50000] 	 Loss: 0.000292247 (0.000275511) 	 Physics Loss: 0.000062929 (0.000051089) 	 Data Loss: 0.000207276 (0.000195997) 	 BC Loss: 0.000022042 (0.000028425)
Iteration: [ 19001/ 50000] 	 Loss: 0.000322576 (0.000273388) 	 Physics Loss: 0.000074615 (0.000057476) 	 Data Loss: 0.000218094 (0.000187935) 	 BC Loss: 0.000029867 (0.000027977)
Iteration: [ 20001/ 50000] 	 Loss: 0.000279427 (0.000258667) 	 Physics Loss: 0.000063707 (0.000045395) 	 Data Loss: 0.000195051 (0.000185922) 	 BC Loss: 0.000020668 (0.000027349)
Iteration: [ 21001/ 50000] 	 Loss: 0.000326586 (0.000258310) 	 Physics Loss: 0.000054800 (0.000048627) 	 Data Loss: 0.000249460 (0.000183803) 	 BC Loss: 0.000022325 (0.000025880)
Iteration: [ 22001/ 50000] 	 Loss: 0.000352667 (0.000270107) 	 Physics Loss: 0.000099547 (0.000059879) 	 Data Loss: 0.000224142 (0.000183791) 	 BC Loss: 0.000028978 (0.000026438)
Iteration: [ 23001/ 50000] 	 Loss: 0.000232196 (0.000234852) 	 Physics Loss: 0.000021778 (0.000037233) 	 Data Loss: 0.000188332 (0.000173101) 	 BC Loss: 0.000022086 (0.000024518)
Iteration: [ 24001/ 50000] 	 Loss: 0.000273704 (0.000230993) 	 Physics Loss: 0.000046370 (0.000035499) 	 Data Loss: 0.000201716 (0.000171023) 	 BC Loss: 0.000025617 (0.000024471)
Iteration: [ 25001/ 50000] 	 Loss: 0.000273417 (0.000234687) 	 Physics Loss: 0.000043496 (0.000049873) 	 Data Loss: 0.000199412 (0.000160869) 	 BC Loss: 0.000030509 (0.000023946)
Iteration: [ 26001/ 50000] 	 Loss: 0.000189405 (0.000242062) 	 Physics Loss: 0.000026152 (0.000046626) 	 Data Loss: 0.000144086 (0.000171613) 	 BC Loss: 0.000019166 (0.000023823)
Iteration: [ 27001/ 50000] 	 Loss: 0.000197575 (0.000218177) 	 Physics Loss: 0.000027417 (0.000036352) 	 Data Loss: 0.000153420 (0.000160356) 	 BC Loss: 0.000016738 (0.000021469)
Iteration: [ 28001/ 50000] 	 Loss: 0.000212760 (0.000231312) 	 Physics Loss: 0.000036409 (0.000046207) 	 Data Loss: 0.000158731 (0.000162394) 	 BC Loss: 0.000017620 (0.000022711)
Iteration: [ 29001/ 50000] 	 Loss: 0.000253997 (0.000220903) 	 Physics Loss: 0.000050729 (0.000036871) 	 Data Loss: 0.000186294 (0.000162614) 	 BC Loss: 0.000016974 (0.000021418)
Iteration: [ 30001/ 50000] 	 Loss: 0.000188432 (0.000198674) 	 Physics Loss: 0.000026784 (0.000026546) 	 Data Loss: 0.000145026 (0.000150229) 	 BC Loss: 0.000016621 (0.000021899)
Iteration: [ 31001/ 50000] 	 Loss: 0.000183548 (0.000210124) 	 Physics Loss: 0.000038238 (0.000029414) 	 Data Loss: 0.000127127 (0.000159928) 	 BC Loss: 0.000018183 (0.000020782)
Iteration: [ 32001/ 50000] 	 Loss: 0.000213935 (0.000220328) 	 Physics Loss: 0.000038426 (0.000040472) 	 Data Loss: 0.000154407 (0.000155267) 	 BC Loss: 0.000021102 (0.000024589)
Iteration: [ 33001/ 50000] 	 Loss: 0.000240946 (0.000201763) 	 Physics Loss: 0.000034044 (0.000028193) 	 Data Loss: 0.000187842 (0.000153407) 	 BC Loss: 0.000019061 (0.000020163)
Iteration: [ 34001/ 50000] 	 Loss: 0.000180558 (0.000198454) 	 Physics Loss: 0.000036186 (0.000024415) 	 Data Loss: 0.000129186 (0.000153794) 	 BC Loss: 0.000015186 (0.000020244)
Iteration: [ 35001/ 50000] 	 Loss: 0.000196057 (0.000198817) 	 Physics Loss: 0.000031524 (0.000031346) 	 Data Loss: 0.000142046 (0.000145053) 	 BC Loss: 0.000022486 (0.000022418)
Iteration: [ 36001/ 50000] 	 Loss: 0.000172539 (0.000212455) 	 Physics Loss: 0.000027912 (0.000043614) 	 Data Loss: 0.000127215 (0.000148391) 	 BC Loss: 0.000017412 (0.000020450)
Iteration: [ 37001/ 50000] 	 Loss: 0.000203403 (0.000223040) 	 Physics Loss: 0.000029162 (0.000044606) 	 Data Loss: 0.000157518 (0.000156260) 	 BC Loss: 0.000016724 (0.000022174)
Iteration: [ 38001/ 50000] 	 Loss: 0.000165642 (0.000199933) 	 Physics Loss: 0.000016988 (0.000027970) 	 Data Loss: 0.000132800 (0.000152079) 	 BC Loss: 0.000015854 (0.000019884)
Iteration: [ 39001/ 50000] 	 Loss: 0.000200040 (0.000192632) 	 Physics Loss: 0.000032169 (0.000023697) 	 Data Loss: 0.000146814 (0.000149103) 	 BC Loss: 0.000021057 (0.000019832)
Iteration: [ 40001/ 50000] 	 Loss: 0.000242139 (0.000201814) 	 Physics Loss: 0.000030695 (0.000035623) 	 Data Loss: 0.000194812 (0.000146296) 	 BC Loss: 0.000016631 (0.000019895)
Iteration: [ 41001/ 50000] 	 Loss: 0.000218949 (0.000184925) 	 Physics Loss: 0.000025431 (0.000023880) 	 Data Loss: 0.000173915 (0.000142177) 	 BC Loss: 0.000019603 (0.000018869)
Iteration: [ 42001/ 50000] 	 Loss: 0.000237525 (0.000191711) 	 Physics Loss: 0.000036015 (0.000026730) 	 Data Loss: 0.000181924 (0.000145566) 	 BC Loss: 0.000019587 (0.000019415)
Iteration: [ 43001/ 50000] 	 Loss: 0.000206539 (0.000202257) 	 Physics Loss: 0.000032655 (0.000030362) 	 Data Loss: 0.000150564 (0.000151022) 	 BC Loss: 0.000023321 (0.000020873)
Iteration: [ 44001/ 50000] 	 Loss: 0.000171327 (0.000188421) 	 Physics Loss: 0.000017162 (0.000029008) 	 Data Loss: 0.000136253 (0.000141201) 	 BC Loss: 0.000017912 (0.000018212)
Iteration: [ 45001/ 50000] 	 Loss: 0.000139038 (0.000190958) 	 Physics Loss: 0.000021031 (0.000032764) 	 Data Loss: 0.000098114 (0.000138178) 	 BC Loss: 0.000019893 (0.000020017)
Iteration: [ 46001/ 50000] 	 Loss: 0.000187017 (0.000190159) 	 Physics Loss: 0.000028397 (0.000024421) 	 Data Loss: 0.000137991 (0.000145876) 	 BC Loss: 0.000020628 (0.000019862)
Iteration: [ 47001/ 50000] 	 Loss: 0.000175257 (0.000185841) 	 Physics Loss: 0.000035563 (0.000025308) 	 Data Loss: 0.000114868 (0.000141361) 	 BC Loss: 0.000024826 (0.000019171)
Iteration: [ 48001/ 50000] 	 Loss: 0.000154686 (0.000186013) 	 Physics Loss: 0.000016418 (0.000025168) 	 Data Loss: 0.000115060 (0.000141419) 	 BC Loss: 0.000023208 (0.000019426)
Iteration: [ 49001/ 50000] 	 Loss: 0.000184697 (0.000183567) 	 Physics Loss: 0.000018286 (0.000024546) 	 Data Loss: 0.000146968 (0.000139627) 	 BC Loss: 0.000019443 (0.000019394)

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
Julia Version 1.13.4
Commit c1ac32ab733 (2026-11-13 17:30 UTC)
Build Info:
  Official https://julialang.org release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 4 × Intel(R) Xeon(R) 6975P-C
  WORD_SIZE: 64
  LLVM: libLLVM-20.1.8 (ORCJIT, graniterapids)
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
