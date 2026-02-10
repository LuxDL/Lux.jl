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
Iteration: [     1/ 50000] 	 Loss: 20.523933411 (20.523933411) 	 Physics Loss: 16.931318283 (16.931318283) 	 Data Loss: 2.007483006 (2.007483006) 	 BC Loss: 1.585133076 (1.585133076)
Iteration: [  1001/ 50000] 	 Loss: 0.017368644 (0.019241152) 	 Physics Loss: 0.000384364 (0.000523634) 	 Data Loss: 0.005318503 (0.007538577) 	 BC Loss: 0.011665777 (0.011178941)
Iteration: [  2001/ 50000] 	 Loss: 0.015431624 (0.018665722) 	 Physics Loss: 0.001248538 (0.001662093) 	 Data Loss: 0.004322519 (0.006408238) 	 BC Loss: 0.009860568 (0.010595392)
Iteration: [  3001/ 50000] 	 Loss: 0.015749734 (0.015215968) 	 Physics Loss: 0.000569924 (0.001279042) 	 Data Loss: 0.004014875 (0.004232418) 	 BC Loss: 0.011164935 (0.009704508)
Iteration: [  4001/ 50000] 	 Loss: 0.009720307 (0.008717594) 	 Physics Loss: 0.002388242 (0.003383865) 	 Data Loss: 0.003175722 (0.002104846) 	 BC Loss: 0.004156343 (0.003228884)
Iteration: [  5001/ 50000] 	 Loss: 0.004859789 (0.005661367) 	 Physics Loss: 0.002144819 (0.002560034) 	 Data Loss: 0.001710090 (0.001477939) 	 BC Loss: 0.001004881 (0.001623392)
Iteration: [  6001/ 50000] 	 Loss: 0.000984606 (0.001235229) 	 Physics Loss: 0.000249066 (0.000298040) 	 Data Loss: 0.000550467 (0.000716767) 	 BC Loss: 0.000185073 (0.000220423)
Iteration: [  7001/ 50000] 	 Loss: 0.001283327 (0.000931201) 	 Physics Loss: 0.000264050 (0.000300163) 	 Data Loss: 0.000923903 (0.000498354) 	 BC Loss: 0.000095375 (0.000132683)
Iteration: [  8001/ 50000] 	 Loss: 0.000837420 (0.000739828) 	 Physics Loss: 0.000461798 (0.000282098) 	 Data Loss: 0.000302047 (0.000380150) 	 BC Loss: 0.000073576 (0.000077580)
Iteration: [  9001/ 50000] 	 Loss: 0.002855566 (0.003176655) 	 Physics Loss: 0.001635206 (0.002080740) 	 Data Loss: 0.000779388 (0.000571454) 	 BC Loss: 0.000440972 (0.000524461)
Iteration: [ 10001/ 50000] 	 Loss: 0.000657131 (0.000839250) 	 Physics Loss: 0.000236588 (0.000455914) 	 Data Loss: 0.000331560 (0.000310165) 	 BC Loss: 0.000088983 (0.000073171)
Iteration: [ 11001/ 50000] 	 Loss: 0.000386514 (0.000371801) 	 Physics Loss: 0.000162340 (0.000067056) 	 Data Loss: 0.000182277 (0.000268991) 	 BC Loss: 0.000041897 (0.000035754)
Iteration: [ 12001/ 50000] 	 Loss: 0.000269073 (0.000349861) 	 Physics Loss: 0.000056341 (0.000066983) 	 Data Loss: 0.000172510 (0.000246374) 	 BC Loss: 0.000040223 (0.000036503)
Iteration: [ 13001/ 50000] 	 Loss: 0.000302106 (0.000322459) 	 Physics Loss: 0.000064594 (0.000065738) 	 Data Loss: 0.000201766 (0.000224176) 	 BC Loss: 0.000035746 (0.000032545)
Iteration: [ 14001/ 50000] 	 Loss: 0.000376156 (0.000328141) 	 Physics Loss: 0.000063470 (0.000066942) 	 Data Loss: 0.000281983 (0.000232647) 	 BC Loss: 0.000030702 (0.000028553)
Iteration: [ 15001/ 50000] 	 Loss: 0.000240459 (0.000284583) 	 Physics Loss: 0.000046128 (0.000056029) 	 Data Loss: 0.000164009 (0.000198332) 	 BC Loss: 0.000030322 (0.000030222)
Iteration: [ 16001/ 50000] 	 Loss: 0.000223565 (0.000286340) 	 Physics Loss: 0.000049468 (0.000059276) 	 Data Loss: 0.000141453 (0.000199299) 	 BC Loss: 0.000032644 (0.000027765)
Iteration: [ 17001/ 50000] 	 Loss: 0.000409930 (0.000304472) 	 Physics Loss: 0.000081523 (0.000072341) 	 Data Loss: 0.000304865 (0.000202506) 	 BC Loss: 0.000023541 (0.000029624)
Iteration: [ 18001/ 50000] 	 Loss: 0.000219687 (0.000283131) 	 Physics Loss: 0.000049917 (0.000060066) 	 Data Loss: 0.000138954 (0.000195451) 	 BC Loss: 0.000030815 (0.000027614)
Iteration: [ 19001/ 50000] 	 Loss: 0.000214799 (0.000274383) 	 Physics Loss: 0.000058947 (0.000052531) 	 Data Loss: 0.000136248 (0.000196898) 	 BC Loss: 0.000019604 (0.000024953)
Iteration: [ 20001/ 50000] 	 Loss: 0.000299889 (0.000261703) 	 Physics Loss: 0.000046930 (0.000056584) 	 Data Loss: 0.000233062 (0.000179937) 	 BC Loss: 0.000019897 (0.000025182)
Iteration: [ 21001/ 50000] 	 Loss: 0.000296875 (0.000240923) 	 Physics Loss: 0.000051274 (0.000047316) 	 Data Loss: 0.000221608 (0.000170391) 	 BC Loss: 0.000023993 (0.000023216)
Iteration: [ 22001/ 50000] 	 Loss: 0.000164491 (0.000243065) 	 Physics Loss: 0.000030805 (0.000053477) 	 Data Loss: 0.000108010 (0.000166853) 	 BC Loss: 0.000025676 (0.000022735)
Iteration: [ 23001/ 50000] 	 Loss: 0.000232463 (0.000275181) 	 Physics Loss: 0.000042860 (0.000074040) 	 Data Loss: 0.000165830 (0.000176594) 	 BC Loss: 0.000023772 (0.000024547)
Iteration: [ 24001/ 50000] 	 Loss: 0.000293330 (0.000257174) 	 Physics Loss: 0.000056128 (0.000061016) 	 Data Loss: 0.000217888 (0.000170106) 	 BC Loss: 0.000019314 (0.000026052)
Iteration: [ 25001/ 50000] 	 Loss: 0.000197672 (0.000226531) 	 Physics Loss: 0.000034642 (0.000036983) 	 Data Loss: 0.000143347 (0.000167836) 	 BC Loss: 0.000019684 (0.000021712)
Iteration: [ 26001/ 50000] 	 Loss: 0.000219556 (0.000242583) 	 Physics Loss: 0.000047353 (0.000055331) 	 Data Loss: 0.000154235 (0.000164076) 	 BC Loss: 0.000017968 (0.000023176)
Iteration: [ 27001/ 50000] 	 Loss: 0.000228742 (0.000228722) 	 Physics Loss: 0.000050512 (0.000042543) 	 Data Loss: 0.000154744 (0.000164306) 	 BC Loss: 0.000023487 (0.000021873)
Iteration: [ 28001/ 50000] 	 Loss: 0.000209967 (0.000221861) 	 Physics Loss: 0.000040376 (0.000042958) 	 Data Loss: 0.000151476 (0.000157371) 	 BC Loss: 0.000018115 (0.000021531)
Iteration: [ 29001/ 50000] 	 Loss: 0.000193407 (0.000226654) 	 Physics Loss: 0.000031548 (0.000049278) 	 Data Loss: 0.000135479 (0.000156947) 	 BC Loss: 0.000026380 (0.000020429)
Iteration: [ 30001/ 50000] 	 Loss: 0.000215555 (0.000228199) 	 Physics Loss: 0.000036278 (0.000045663) 	 Data Loss: 0.000155981 (0.000161072) 	 BC Loss: 0.000023296 (0.000021465)
Iteration: [ 31001/ 50000] 	 Loss: 0.000267548 (0.000229684) 	 Physics Loss: 0.000032738 (0.000049684) 	 Data Loss: 0.000214134 (0.000158689) 	 BC Loss: 0.000020676 (0.000021311)
Iteration: [ 32001/ 50000] 	 Loss: 0.000209865 (0.000204781) 	 Physics Loss: 0.000043204 (0.000033433) 	 Data Loss: 0.000146945 (0.000151769) 	 BC Loss: 0.000019717 (0.000019580)
Iteration: [ 33001/ 50000] 	 Loss: 0.000173422 (0.000197627) 	 Physics Loss: 0.000019830 (0.000031272) 	 Data Loss: 0.000132011 (0.000146727) 	 BC Loss: 0.000021581 (0.000019628)
Iteration: [ 34001/ 50000] 	 Loss: 0.000195550 (0.000200745) 	 Physics Loss: 0.000031429 (0.000035648) 	 Data Loss: 0.000141983 (0.000145135) 	 BC Loss: 0.000022139 (0.000019962)
Iteration: [ 35001/ 50000] 	 Loss: 0.000134375 (0.000209997) 	 Physics Loss: 0.000019678 (0.000043786) 	 Data Loss: 0.000092189 (0.000146878) 	 BC Loss: 0.000022507 (0.000019333)
Iteration: [ 36001/ 50000] 	 Loss: 0.000181370 (0.000201183) 	 Physics Loss: 0.000042073 (0.000035633) 	 Data Loss: 0.000115895 (0.000146339) 	 BC Loss: 0.000023402 (0.000019210)
Iteration: [ 37001/ 50000] 	 Loss: 0.000293229 (0.000189756) 	 Physics Loss: 0.000058758 (0.000025711) 	 Data Loss: 0.000216189 (0.000145108) 	 BC Loss: 0.000018282 (0.000018937)
Iteration: [ 38001/ 50000] 	 Loss: 0.000246447 (0.000217078) 	 Physics Loss: 0.000033519 (0.000046140) 	 Data Loss: 0.000189671 (0.000152419) 	 BC Loss: 0.000023257 (0.000018519)
Iteration: [ 39001/ 50000] 	 Loss: 0.000177160 (0.000196670) 	 Physics Loss: 0.000037388 (0.000032300) 	 Data Loss: 0.000123830 (0.000145750) 	 BC Loss: 0.000015942 (0.000018620)
Iteration: [ 40001/ 50000] 	 Loss: 0.000166135 (0.000188288) 	 Physics Loss: 0.000019232 (0.000024995) 	 Data Loss: 0.000125661 (0.000143894) 	 BC Loss: 0.000021242 (0.000019400)
Iteration: [ 41001/ 50000] 	 Loss: 0.000159792 (0.000193006) 	 Physics Loss: 0.000022440 (0.000028827) 	 Data Loss: 0.000120105 (0.000146391) 	 BC Loss: 0.000017247 (0.000017789)
Iteration: [ 42001/ 50000] 	 Loss: 0.000177615 (0.000191759) 	 Physics Loss: 0.000027017 (0.000028250) 	 Data Loss: 0.000134263 (0.000145198) 	 BC Loss: 0.000016335 (0.000018311)
Iteration: [ 43001/ 50000] 	 Loss: 0.000188687 (0.000196289) 	 Physics Loss: 0.000024537 (0.000030418) 	 Data Loss: 0.000150235 (0.000145287) 	 BC Loss: 0.000013915 (0.000020584)
Iteration: [ 44001/ 50000] 	 Loss: 0.000173374 (0.000188091) 	 Physics Loss: 0.000016809 (0.000026137) 	 Data Loss: 0.000142598 (0.000141800) 	 BC Loss: 0.000013967 (0.000020153)
Iteration: [ 45001/ 50000] 	 Loss: 0.000274685 (0.000224483) 	 Physics Loss: 0.000040260 (0.000049421) 	 Data Loss: 0.000206662 (0.000152599) 	 BC Loss: 0.000027762 (0.000022464)
Iteration: [ 46001/ 50000] 	 Loss: 0.000198185 (0.000179626) 	 Physics Loss: 0.000025413 (0.000023982) 	 Data Loss: 0.000153957 (0.000138125) 	 BC Loss: 0.000018815 (0.000017519)
Iteration: [ 47001/ 50000] 	 Loss: 0.000172053 (0.000188898) 	 Physics Loss: 0.000019933 (0.000031009) 	 Data Loss: 0.000135304 (0.000139323) 	 BC Loss: 0.000016816 (0.000018566)
Iteration: [ 48001/ 50000] 	 Loss: 0.000184741 (0.000185369) 	 Physics Loss: 0.000025975 (0.000029012) 	 Data Loss: 0.000139943 (0.000138431) 	 BC Loss: 0.000018824 (0.000017926)
Iteration: [ 49001/ 50000] 	 Loss: 0.000166072 (0.000191901) 	 Physics Loss: 0.000023222 (0.000033150) 	 Data Loss: 0.000122174 (0.000140722) 	 BC Loss: 0.000020676 (0.000018028)

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
Julia Version 1.12.4
Commit 01a2eadb047 (2026-01-06 16:56 UTC)
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
