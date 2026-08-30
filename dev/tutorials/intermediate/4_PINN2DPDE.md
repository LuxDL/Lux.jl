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
Iteration: [  1001/ 50000] 	 Loss: 0.017368633 (0.019241147) 	 Physics Loss: 0.000384352 (0.000523631) 	 Data Loss: 0.005318501 (0.007538573) 	 BC Loss: 0.011665781 (0.011178940)
Iteration: [  2001/ 50000] 	 Loss: 0.015431664 (0.018665710) 	 Physics Loss: 0.001248588 (0.001662086) 	 Data Loss: 0.004322519 (0.006408236) 	 BC Loss: 0.009860558 (0.010595387)
Iteration: [  3001/ 50000] 	 Loss: 0.015749730 (0.015216020) 	 Physics Loss: 0.000569911 (0.001279066) 	 Data Loss: 0.004014889 (0.004232434) 	 BC Loss: 0.011164930 (0.009704520)
Iteration: [  4001/ 50000] 	 Loss: 0.009721411 (0.008720266) 	 Physics Loss: 0.002389135 (0.003386307) 	 Data Loss: 0.003174899 (0.002104989) 	 BC Loss: 0.004157377 (0.003228969)
Iteration: [  5001/ 50000] 	 Loss: 0.004499406 (0.004708527) 	 Physics Loss: 0.001959276 (0.001955591) 	 Data Loss: 0.001685009 (0.001497321) 	 BC Loss: 0.000855121 (0.001255615)
Iteration: [  6001/ 50000] 	 Loss: 0.001065215 (0.001272577) 	 Physics Loss: 0.000281185 (0.000287863) 	 Data Loss: 0.000585776 (0.000744654) 	 BC Loss: 0.000198253 (0.000240060)
Iteration: [  7001/ 50000] 	 Loss: 0.001347864 (0.000929980) 	 Physics Loss: 0.000297512 (0.000298550) 	 Data Loss: 0.000953921 (0.000510000) 	 BC Loss: 0.000096431 (0.000121430)
Iteration: [  8001/ 50000] 	 Loss: 0.001117045 (0.000741427) 	 Physics Loss: 0.000746706 (0.000274790) 	 Data Loss: 0.000305004 (0.000386474) 	 BC Loss: 0.000065335 (0.000080163)
Iteration: [  9001/ 50000] 	 Loss: 0.001055349 (0.001118247) 	 Physics Loss: 0.000357186 (0.000671388) 	 Data Loss: 0.000604138 (0.000345194) 	 BC Loss: 0.000094025 (0.000101665)
Iteration: [ 10001/ 50000] 	 Loss: 0.000577895 (0.000586777) 	 Physics Loss: 0.000227950 (0.000239104) 	 Data Loss: 0.000296419 (0.000296933) 	 BC Loss: 0.000053526 (0.000050740)
Iteration: [ 11001/ 50000] 	 Loss: 0.000387099 (0.000371386) 	 Physics Loss: 0.000160282 (0.000064884) 	 Data Loss: 0.000181874 (0.000270512) 	 BC Loss: 0.000044943 (0.000035990)
Iteration: [ 12001/ 50000] 	 Loss: 0.000264985 (0.000347079) 	 Physics Loss: 0.000050845 (0.000063801) 	 Data Loss: 0.000172765 (0.000247937) 	 BC Loss: 0.000041375 (0.000035341)
Iteration: [ 13001/ 50000] 	 Loss: 0.000303605 (0.000328298) 	 Physics Loss: 0.000061343 (0.000068644) 	 Data Loss: 0.000208084 (0.000226532) 	 BC Loss: 0.000034178 (0.000033121)
Iteration: [ 14001/ 50000] 	 Loss: 0.000384829 (0.000331977) 	 Physics Loss: 0.000068277 (0.000069249) 	 Data Loss: 0.000282153 (0.000234252) 	 BC Loss: 0.000034399 (0.000028476)
Iteration: [ 15001/ 50000] 	 Loss: 0.000239852 (0.000288137) 	 Physics Loss: 0.000043116 (0.000057610) 	 Data Loss: 0.000166922 (0.000199768) 	 BC Loss: 0.000029814 (0.000030758)
Iteration: [ 16001/ 50000] 	 Loss: 0.000223553 (0.000286669) 	 Physics Loss: 0.000050308 (0.000056936) 	 Data Loss: 0.000142008 (0.000201499) 	 BC Loss: 0.000031237 (0.000028234)
Iteration: [ 17001/ 50000] 	 Loss: 0.000421828 (0.000296886) 	 Physics Loss: 0.000095925 (0.000067681) 	 Data Loss: 0.000300998 (0.000201824) 	 BC Loss: 0.000024905 (0.000027381)
Iteration: [ 18001/ 50000] 	 Loss: 0.000211644 (0.000281161) 	 Physics Loss: 0.000038092 (0.000057524) 	 Data Loss: 0.000140487 (0.000195981) 	 BC Loss: 0.000033065 (0.000027656)
Iteration: [ 19001/ 50000] 	 Loss: 0.000211127 (0.000277151) 	 Physics Loss: 0.000053733 (0.000055449) 	 Data Loss: 0.000138345 (0.000197246) 	 BC Loss: 0.000019049 (0.000024456)
Iteration: [ 20001/ 50000] 	 Loss: 0.000310542 (0.000249381) 	 Physics Loss: 0.000061448 (0.000046612) 	 Data Loss: 0.000229472 (0.000180468) 	 BC Loss: 0.000019622 (0.000022300)
Iteration: [ 21001/ 50000] 	 Loss: 0.000297450 (0.000248106) 	 Physics Loss: 0.000059162 (0.000053576) 	 Data Loss: 0.000216021 (0.000171196) 	 BC Loss: 0.000022266 (0.000023334)
Iteration: [ 22001/ 50000] 	 Loss: 0.000164182 (0.000235557) 	 Physics Loss: 0.000028181 (0.000045489) 	 Data Loss: 0.000109619 (0.000167632) 	 BC Loss: 0.000026382 (0.000022436)
Iteration: [ 23001/ 50000] 	 Loss: 0.000232555 (0.000246731) 	 Physics Loss: 0.000037212 (0.000047894) 	 Data Loss: 0.000171760 (0.000177439) 	 BC Loss: 0.000023583 (0.000021399)
Iteration: [ 24001/ 50000] 	 Loss: 0.000292702 (0.000249951) 	 Physics Loss: 0.000055334 (0.000055020) 	 Data Loss: 0.000219019 (0.000170300) 	 BC Loss: 0.000018350 (0.000024631)
Iteration: [ 25001/ 50000] 	 Loss: 0.000206420 (0.000227030) 	 Physics Loss: 0.000044298 (0.000038203) 	 Data Loss: 0.000143160 (0.000167706) 	 BC Loss: 0.000018962 (0.000021121)
Iteration: [ 26001/ 50000] 	 Loss: 0.000229478 (0.000242765) 	 Physics Loss: 0.000053504 (0.000056142) 	 Data Loss: 0.000154916 (0.000164058) 	 BC Loss: 0.000021058 (0.000022566)
Iteration: [ 27001/ 50000] 	 Loss: 0.000224114 (0.000242052) 	 Physics Loss: 0.000045716 (0.000054974) 	 Data Loss: 0.000152396 (0.000164748) 	 BC Loss: 0.000026002 (0.000022330)
Iteration: [ 28001/ 50000] 	 Loss: 0.000222216 (0.000215397) 	 Physics Loss: 0.000050252 (0.000036081) 	 Data Loss: 0.000154347 (0.000157962) 	 BC Loss: 0.000017617 (0.000021354)
Iteration: [ 29001/ 50000] 	 Loss: 0.000210704 (0.000233809) 	 Physics Loss: 0.000035138 (0.000053348) 	 Data Loss: 0.000143131 (0.000158839) 	 BC Loss: 0.000032434 (0.000021622)
Iteration: [ 30001/ 50000] 	 Loss: 0.000207375 (0.000226769) 	 Physics Loss: 0.000026813 (0.000043914) 	 Data Loss: 0.000155654 (0.000161216) 	 BC Loss: 0.000024908 (0.000021639)
Iteration: [ 31001/ 50000] 	 Loss: 0.000269214 (0.000219518) 	 Physics Loss: 0.000039355 (0.000040133) 	 Data Loss: 0.000209013 (0.000158317) 	 BC Loss: 0.000020846 (0.000021067)
Iteration: [ 32001/ 50000] 	 Loss: 0.000211419 (0.000210957) 	 Physics Loss: 0.000045159 (0.000038796) 	 Data Loss: 0.000146814 (0.000151931) 	 BC Loss: 0.000019446 (0.000020230)
Iteration: [ 33001/ 50000] 	 Loss: 0.000181482 (0.000202039) 	 Physics Loss: 0.000025003 (0.000032754) 	 Data Loss: 0.000134261 (0.000148478) 	 BC Loss: 0.000022218 (0.000020807)
Iteration: [ 34001/ 50000] 	 Loss: 0.000186912 (0.000197588) 	 Physics Loss: 0.000025085 (0.000032541) 	 Data Loss: 0.000141114 (0.000145119) 	 BC Loss: 0.000020713 (0.000019928)
Iteration: [ 35001/ 50000] 	 Loss: 0.000136710 (0.000217637) 	 Physics Loss: 0.000018034 (0.000050786) 	 Data Loss: 0.000097950 (0.000147103) 	 BC Loss: 0.000020726 (0.000019747)
Iteration: [ 36001/ 50000] 	 Loss: 0.000158078 (0.000199009) 	 Physics Loss: 0.000024817 (0.000033946) 	 Data Loss: 0.000114559 (0.000146315) 	 BC Loss: 0.000018701 (0.000018748)
Iteration: [ 37001/ 50000] 	 Loss: 0.000302340 (0.000190165) 	 Physics Loss: 0.000066689 (0.000025736) 	 Data Loss: 0.000217540 (0.000145103) 	 BC Loss: 0.000018111 (0.000019326)
Iteration: [ 38001/ 50000] 	 Loss: 0.000245146 (0.000204957) 	 Physics Loss: 0.000030379 (0.000036335) 	 Data Loss: 0.000189912 (0.000150965) 	 BC Loss: 0.000024854 (0.000017657)
Iteration: [ 39001/ 50000] 	 Loss: 0.000165623 (0.000197576) 	 Physics Loss: 0.000026971 (0.000032796) 	 Data Loss: 0.000122209 (0.000145698) 	 BC Loss: 0.000016443 (0.000019082)
Iteration: [ 40001/ 50000] 	 Loss: 0.000171607 (0.000195268) 	 Physics Loss: 0.000026284 (0.000031357) 	 Data Loss: 0.000123539 (0.000143961) 	 BC Loss: 0.000021784 (0.000019949)
Iteration: [ 41001/ 50000] 	 Loss: 0.000156535 (0.000193990) 	 Physics Loss: 0.000019421 (0.000028832) 	 Data Loss: 0.000119100 (0.000146621) 	 BC Loss: 0.000018014 (0.000018538)
Iteration: [ 42001/ 50000] 	 Loss: 0.000175756 (0.000190178) 	 Physics Loss: 0.000023863 (0.000026718) 	 Data Loss: 0.000135717 (0.000144790) 	 BC Loss: 0.000016176 (0.000018671)
Iteration: [ 43001/ 50000] 	 Loss: 0.000191984 (0.000191988) 	 Physics Loss: 0.000026218 (0.000027192) 	 Data Loss: 0.000150546 (0.000144479) 	 BC Loss: 0.000015221 (0.000020318)
Iteration: [ 44001/ 50000] 	 Loss: 0.000171279 (0.000191033) 	 Physics Loss: 0.000011855 (0.000027447) 	 Data Loss: 0.000144759 (0.000142707) 	 BC Loss: 0.000014665 (0.000020879)
Iteration: [ 45001/ 50000] 	 Loss: 0.000266982 (0.000207141) 	 Physics Loss: 0.000038990 (0.000034863) 	 Data Loss: 0.000201324 (0.000150720) 	 BC Loss: 0.000026669 (0.000021558)
Iteration: [ 46001/ 50000] 	 Loss: 0.000201456 (0.000188971) 	 Physics Loss: 0.000026719 (0.000033167) 	 Data Loss: 0.000155729 (0.000138071) 	 BC Loss: 0.000019007 (0.000017732)
Iteration: [ 47001/ 50000] 	 Loss: 0.000180822 (0.000185797) 	 Physics Loss: 0.000026415 (0.000028996) 	 Data Loss: 0.000137763 (0.000138725) 	 BC Loss: 0.000016644 (0.000018075)
Iteration: [ 48001/ 50000] 	 Loss: 0.000188553 (0.000178357) 	 Physics Loss: 0.000028802 (0.000022570) 	 Data Loss: 0.000141659 (0.000137397) 	 BC Loss: 0.000018092 (0.000018390)
Iteration: [ 49001/ 50000] 	 Loss: 0.000172877 (0.000196456) 	 Physics Loss: 0.000030594 (0.000037507) 	 Data Loss: 0.000120721 (0.000140691) 	 BC Loss: 0.000021562 (0.000018257)

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
  CPU: 4 × Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, icelake-server)
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
