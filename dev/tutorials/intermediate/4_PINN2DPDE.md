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
Iteration: [     1/ 50000] 	 Loss: 20.523931503 (20.523931503) 	 Physics Loss: 16.931316376 (16.931316376) 	 Data Loss: 2.007482767 (2.007482767) 	 BC Loss: 1.585133195 (1.585133195)
Iteration: [  1001/ 50000] 	 Loss: 0.017368628 (0.019241145) 	 Physics Loss: 0.000384354 (0.000523634) 	 Data Loss: 0.005318501 (0.007538572) 	 BC Loss: 0.011665773 (0.011178938)
Iteration: [  2001/ 50000] 	 Loss: 0.015431687 (0.018665681) 	 Physics Loss: 0.001248603 (0.001662047) 	 Data Loss: 0.004322523 (0.006408238) 	 BC Loss: 0.009860562 (0.010595394)
Iteration: [  3001/ 50000] 	 Loss: 0.015749767 (0.015216094) 	 Physics Loss: 0.000569927 (0.001279120) 	 Data Loss: 0.004014903 (0.004232446) 	 BC Loss: 0.011164938 (0.009704530)
Iteration: [  4001/ 50000] 	 Loss: 0.009716226 (0.008706219) 	 Physics Loss: 0.002386823 (0.003374664) 	 Data Loss: 0.003176760 (0.002104126) 	 BC Loss: 0.004152643 (0.003227429)
Iteration: [  5001/ 50000] 	 Loss: 0.003886510 (0.005067998) 	 Physics Loss: 0.001579347 (0.002175978) 	 Data Loss: 0.001516973 (0.001368281) 	 BC Loss: 0.000790190 (0.001523739)
Iteration: [  6001/ 50000] 	 Loss: 0.001034419 (0.001255641) 	 Physics Loss: 0.000291792 (0.000311166) 	 Data Loss: 0.000561496 (0.000719778) 	 BC Loss: 0.000181132 (0.000224697)
Iteration: [  7001/ 50000] 	 Loss: 0.001385733 (0.000930352) 	 Physics Loss: 0.000333133 (0.000301294) 	 Data Loss: 0.000951870 (0.000498431) 	 BC Loss: 0.000100731 (0.000130627)
Iteration: [  8001/ 50000] 	 Loss: 0.001724537 (0.000995430) 	 Physics Loss: 0.001351545 (0.000478673) 	 Data Loss: 0.000303639 (0.000409743) 	 BC Loss: 0.000069353 (0.000107014)
Iteration: [  9001/ 50000] 	 Loss: 0.001866363 (0.002031762) 	 Physics Loss: 0.000962198 (0.001362235) 	 Data Loss: 0.000725944 (0.000449483) 	 BC Loss: 0.000178221 (0.000220044)
Iteration: [ 10001/ 50000] 	 Loss: 0.000635277 (0.000728421) 	 Physics Loss: 0.000261422 (0.000358250) 	 Data Loss: 0.000295392 (0.000308264) 	 BC Loss: 0.000078463 (0.000061907)
Iteration: [ 11001/ 50000] 	 Loss: 0.000383008 (0.000377838) 	 Physics Loss: 0.000153039 (0.000066602) 	 Data Loss: 0.000184805 (0.000273820) 	 BC Loss: 0.000045163 (0.000037416)
Iteration: [ 12001/ 50000] 	 Loss: 0.000261521 (0.000352937) 	 Physics Loss: 0.000048594 (0.000064999) 	 Data Loss: 0.000170984 (0.000251046) 	 BC Loss: 0.000041943 (0.000036892)
Iteration: [ 13001/ 50000] 	 Loss: 0.000307155 (0.000331815) 	 Physics Loss: 0.000063350 (0.000068230) 	 Data Loss: 0.000209438 (0.000229331) 	 BC Loss: 0.000034366 (0.000034255)
Iteration: [ 14001/ 50000] 	 Loss: 0.000383858 (0.000333428) 	 Physics Loss: 0.000058910 (0.000066048) 	 Data Loss: 0.000290878 (0.000238018) 	 BC Loss: 0.000034070 (0.000029362)
Iteration: [ 15001/ 50000] 	 Loss: 0.000239199 (0.000289081) 	 Physics Loss: 0.000039767 (0.000055478) 	 Data Loss: 0.000169393 (0.000202013) 	 BC Loss: 0.000030040 (0.000031589)
Iteration: [ 16001/ 50000] 	 Loss: 0.000223157 (0.000291498) 	 Physics Loss: 0.000050619 (0.000058149) 	 Data Loss: 0.000141494 (0.000203895) 	 BC Loss: 0.000031044 (0.000029453)
Iteration: [ 17001/ 50000] 	 Loss: 0.000424605 (0.000300286) 	 Physics Loss: 0.000094563 (0.000067444) 	 Data Loss: 0.000306057 (0.000204684) 	 BC Loss: 0.000023986 (0.000028158)
Iteration: [ 18001/ 50000] 	 Loss: 0.000211657 (0.000274348) 	 Physics Loss: 0.000039484 (0.000050938) 	 Data Loss: 0.000141258 (0.000197313) 	 BC Loss: 0.000030915 (0.000026097)
Iteration: [ 19001/ 50000] 	 Loss: 0.000215204 (0.000271044) 	 Physics Loss: 0.000057041 (0.000047980) 	 Data Loss: 0.000138520 (0.000198956) 	 BC Loss: 0.000019642 (0.000024109)
Iteration: [ 20001/ 50000] 	 Loss: 0.000308061 (0.000259510) 	 Physics Loss: 0.000054062 (0.000053414) 	 Data Loss: 0.000234109 (0.000182513) 	 BC Loss: 0.000019890 (0.000023583)
Iteration: [ 21001/ 50000] 	 Loss: 0.000300082 (0.000242999) 	 Physics Loss: 0.000056181 (0.000048348) 	 Data Loss: 0.000222399 (0.000171826) 	 BC Loss: 0.000021501 (0.000022825)
Iteration: [ 22001/ 50000] 	 Loss: 0.000167973 (0.000235056) 	 Physics Loss: 0.000032013 (0.000044670) 	 Data Loss: 0.000109637 (0.000168704) 	 BC Loss: 0.000026323 (0.000021682)
Iteration: [ 23001/ 50000] 	 Loss: 0.000229293 (0.000245322) 	 Physics Loss: 0.000036264 (0.000046493) 	 Data Loss: 0.000169247 (0.000177739) 	 BC Loss: 0.000023783 (0.000021090)
Iteration: [ 24001/ 50000] 	 Loss: 0.000289776 (0.000246447) 	 Physics Loss: 0.000047102 (0.000049579) 	 Data Loss: 0.000222362 (0.000171563) 	 BC Loss: 0.000020312 (0.000025306)
Iteration: [ 25001/ 50000] 	 Loss: 0.000209445 (0.000223173) 	 Physics Loss: 0.000046892 (0.000034051) 	 Data Loss: 0.000143755 (0.000168449) 	 BC Loss: 0.000018798 (0.000020673)
Iteration: [ 26001/ 50000] 	 Loss: 0.000231140 (0.000248499) 	 Physics Loss: 0.000053100 (0.000059397) 	 Data Loss: 0.000157614 (0.000164726) 	 BC Loss: 0.000020426 (0.000024376)
Iteration: [ 27001/ 50000] 	 Loss: 0.000222032 (0.000238464) 	 Physics Loss: 0.000044071 (0.000051872) 	 Data Loss: 0.000152020 (0.000165182) 	 BC Loss: 0.000025941 (0.000021410)
Iteration: [ 28001/ 50000] 	 Loss: 0.000213803 (0.000217214) 	 Physics Loss: 0.000043811 (0.000038101) 	 Data Loss: 0.000154321 (0.000158097) 	 BC Loss: 0.000015670 (0.000021016)
Iteration: [ 29001/ 50000] 	 Loss: 0.000218759 (0.000233276) 	 Physics Loss: 0.000049815 (0.000054533) 	 Data Loss: 0.000139171 (0.000158301) 	 BC Loss: 0.000029773 (0.000020442)
Iteration: [ 30001/ 50000] 	 Loss: 0.000197796 (0.000216997) 	 Physics Loss: 0.000022928 (0.000036634) 	 Data Loss: 0.000155644 (0.000160307) 	 BC Loss: 0.000019224 (0.000020056)
Iteration: [ 31001/ 50000] 	 Loss: 0.000272233 (0.000217908) 	 Physics Loss: 0.000037594 (0.000039389) 	 Data Loss: 0.000213478 (0.000158420) 	 BC Loss: 0.000021160 (0.000020100)
Iteration: [ 32001/ 50000] 	 Loss: 0.000205471 (0.000203662) 	 Physics Loss: 0.000036859 (0.000032520) 	 Data Loss: 0.000148862 (0.000151868) 	 BC Loss: 0.000019750 (0.000019273)
Iteration: [ 33001/ 50000] 	 Loss: 0.000177065 (0.000194660) 	 Physics Loss: 0.000024218 (0.000027367) 	 Data Loss: 0.000132047 (0.000147741) 	 BC Loss: 0.000020799 (0.000019552)
Iteration: [ 34001/ 50000] 	 Loss: 0.000197646 (0.000198811) 	 Physics Loss: 0.000037049 (0.000035072) 	 Data Loss: 0.000139941 (0.000144311) 	 BC Loss: 0.000020656 (0.000019428)
Iteration: [ 35001/ 50000] 	 Loss: 0.000139140 (0.000205813) 	 Physics Loss: 0.000021988 (0.000040625) 	 Data Loss: 0.000099303 (0.000146617) 	 BC Loss: 0.000017849 (0.000018571)
Iteration: [ 36001/ 50000] 	 Loss: 0.000156492 (0.000198452) 	 Physics Loss: 0.000022780 (0.000034482) 	 Data Loss: 0.000115434 (0.000145928) 	 BC Loss: 0.000018278 (0.000018042)
Iteration: [ 37001/ 50000] 	 Loss: 0.000301301 (0.000190601) 	 Physics Loss: 0.000066999 (0.000027148) 	 Data Loss: 0.000217312 (0.000144643) 	 BC Loss: 0.000016991 (0.000018809)
Iteration: [ 38001/ 50000] 	 Loss: 0.000239282 (0.000194629) 	 Physics Loss: 0.000025748 (0.000027692) 	 Data Loss: 0.000189683 (0.000150262) 	 BC Loss: 0.000023851 (0.000016676)
Iteration: [ 39001/ 50000] 	 Loss: 0.000165712 (0.000192547) 	 Physics Loss: 0.000027288 (0.000029409) 	 Data Loss: 0.000122769 (0.000144881) 	 BC Loss: 0.000015655 (0.000018257)
Iteration: [ 40001/ 50000] 	 Loss: 0.000174853 (0.000192350) 	 Physics Loss: 0.000033125 (0.000030111) 	 Data Loss: 0.000122016 (0.000143671) 	 BC Loss: 0.000019712 (0.000018569)
Iteration: [ 41001/ 50000] 	 Loss: 0.000154217 (0.000189001) 	 Physics Loss: 0.000018828 (0.000025653) 	 Data Loss: 0.000118776 (0.000145906) 	 BC Loss: 0.000016613 (0.000017443)
Iteration: [ 42001/ 50000] 	 Loss: 0.000174628 (0.000189114) 	 Physics Loss: 0.000025195 (0.000025773) 	 Data Loss: 0.000134023 (0.000144920) 	 BC Loss: 0.000015410 (0.000018421)
Iteration: [ 43001/ 50000] 	 Loss: 0.000202906 (0.000193699) 	 Physics Loss: 0.000037348 (0.000029028) 	 Data Loss: 0.000151805 (0.000144529) 	 BC Loss: 0.000013753 (0.000020142)
Iteration: [ 44001/ 50000] 	 Loss: 0.000170139 (0.000194087) 	 Physics Loss: 0.000013604 (0.000030771) 	 Data Loss: 0.000141844 (0.000142223) 	 BC Loss: 0.000014691 (0.000021093)
Iteration: [ 45001/ 50000] 	 Loss: 0.000268400 (0.000214387) 	 Physics Loss: 0.000040327 (0.000041171) 	 Data Loss: 0.000202639 (0.000151532) 	 BC Loss: 0.000025435 (0.000021684)
Iteration: [ 46001/ 50000] 	 Loss: 0.000195847 (0.000179004) 	 Physics Loss: 0.000021735 (0.000024361) 	 Data Loss: 0.000156592 (0.000137531) 	 BC Loss: 0.000017520 (0.000017112)
Iteration: [ 47001/ 50000] 	 Loss: 0.000171102 (0.000182314) 	 Physics Loss: 0.000017443 (0.000026599) 	 Data Loss: 0.000136824 (0.000138374) 	 BC Loss: 0.000016835 (0.000017341)
Iteration: [ 48001/ 50000] 	 Loss: 0.000182627 (0.000178462) 	 Physics Loss: 0.000024490 (0.000023615) 	 Data Loss: 0.000140596 (0.000137281) 	 BC Loss: 0.000017540 (0.000017566)
Iteration: [ 49001/ 50000] 	 Loss: 0.000171221 (0.000194728) 	 Physics Loss: 0.000028240 (0.000035693) 	 Data Loss: 0.000123371 (0.000141317) 	 BC Loss: 0.000019610 (0.000017718)

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
  CPU: 4 × Intel(R) Xeon(R) 6973P-C
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
