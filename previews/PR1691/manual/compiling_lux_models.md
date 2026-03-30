---
url: /previews/PR1691/manual/compiling_lux_models.md
---
# Compiling Lux Models using `Reactant.jl` {#reactant-compilation}

Quoting the Reactant.jl Readme:

> Reactant takes Julia function and compile it into MLIR and run fancy optimizations on top of it, including using EnzymeMLIR for automatic differentiation, and create relevant executables for CPU/GPU/TPU via XLA. It presently operates as a tracing system. Compiled functions will assume the same control flow pattern as was original taken by objects used at compile time, and control flow (e.g. if, for) as well as any type instabilities will be removed. The benefits of this approach is immediately making all such code available for advanced optimization with little developer effort.

```julia
using Lux, Reactant, Enzyme, Random, Zygote
using Functors, Optimisers, Printf
```

::: tip Running on alternate accelerators

`Reactant.set_default_backend("gpu")` sets the default backend to CUDA and `Reactant.set_default_backend("tpu")` sets the default backend to TPU.

:::

::: tip Using the `TrainState` API

If you are using the [`Training.TrainState`](/api/Lux/utilities#Lux.Training.TrainState) API, skip to the [bottom of this page](/manual/compiling_lux_models#compile_lux_model_trainstate) to see how to train the model without any of this boilerplate.

:::

We start by defining a simple MLP model:

```julia
model = Chain(
    Dense(2 => 32, gelu),
    Dense(32 => 32, gelu),
    Dense(32 => 2)
)
ps, st = Lux.setup(Random.default_rng(), model)
```

```ansi
((layer_1 = (weight = Float32[0.9670442 -0.36027783; 0.078672916 0.92788666; … ; -0.65058047 -0.47006413; -0.48801818 -0.6615898], bias = Float32[-0.28780195, -0.23392133, 0.084573634, -0.59277534, -0.6795253, 0.47792822, -0.64850235, -0.55131584, -0.33091125, 0.47174177  …  0.07477753, -0.10521463, -0.45745936, 0.19031122, 0.41613227, 0.47329637, -0.68522483, -0.2834571, 0.0235815, 0.61977077]), layer_2 = (weight = Float32[-0.057887085 -0.14646342 … 0.1019723 0.14663221; 0.10022328 -0.09659223 … 0.25911948 -0.008825431; … ; -0.014519578 -0.01100632 … -0.30112675 -0.17886546; 0.21983564 -0.026677115 … -0.030971587 -0.28283697], bias = Float32[0.095548995, 0.10995198, 0.12209795, -0.14433007, 0.11754602, -0.152131, -0.10584956, 0.09469124, 0.09255884, 0.10044085  …  0.07444663, 0.11096934, 0.13462374, 0.15048876, 0.061646424, 0.004753132, 0.08162795, -0.15708117, 0.029835312, 0.005353872]), layer_3 = (weight = Float32[0.005372945 -0.18356045 … 0.052086722 0.07186686; 0.0067291846 0.020219602 … 0.0688707 -0.1961357], bias = Float32[-0.03542879, -0.041368797])), (layer_1 = NamedTuple(), layer_2 = NamedTuple(), layer_3 = NamedTuple()))
```

We then create a random input and output data:

```julia
x = randn(Float32, 2, 32)
y = x .^ 2
```

We will use [`reactant_device`](/api/Accelerator_Support/MLDataDevices#MLDataDevices.reactant_device) similar to [`gpu_device`](/api/Accelerator_Support/MLDataDevices#MLDataDevices.gpu_device) to move the arrays to `Reactant`.

```julia
const xdev = reactant_device()

x_ra = x |> xdev
y_ra = y |> xdev
ps_ra = ps |> xdev
st_ra = st |> xdev
```

First let's run the model as we would normally:

```julia
pred_lux, _ = model(x, ps, Lux.testmode(st))
```

```ansi
(Float32[0.0158698 0.010564301 … -0.41376624 0.018748913; 0.07865399 0.06953074 … -0.2340262 0.21624331], (layer_1 = NamedTuple(), layer_2 = NamedTuple(), layer_3 = NamedTuple()))
```

To run it using `XLA` we need to compile the model. We can do this using the `Reactant.@compile` macro. Note that the inputs need to be moved to the device using [`reactant_device`](/api/Accelerator_Support/MLDataDevices#MLDataDevices.reactant_device) first.

```julia
model_compiled = @compile model(x_ra, ps_ra, Lux.testmode(st_ra))
```

```ansi
Reactant compiled function Chain{@NamedTuple{layer_1::Dense{typeof(gelu_tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Dense{typeof(gelu_tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_3::Dense{typeof(identity), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing}((layer_1 = Dense(2 => 32, gelu_tanh), layer_2 = Dense(32 => 32, gelu_tanh), layer_3 = Dense(32 => 2)), nothing) (with tag ##Chain{@NamedTuple{layer_1::Dense{typeof(gelu_tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Dense{typeof(gelu_tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_3::Dense{typeof(identity), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing}((layer_1 = Dense(2 => 32, gelu_tanh), layer_2 = Dense(32 => 32, gelu_tanh), layer_3 = Dense(32 => 2)), nothing)_reactant#896579)
```

Now we can test the difference between the results:

```julia
pred_compiled, _ = model_compiled(x_ra, ps_ra, Lux.testmode(st_ra))

pred_lux .- Array(pred_compiled)
```

```ansi
2×32 Matrix{Float32}:
 -3.72529f-9  -3.72529f-9  7.45058f-9  …  0.0         -7.07805f-8
  7.45058f-9   0.0         0.0            4.47035f-8   1.19209f-7
```

The difference is very small as we would expect. Now, let's try to differentiate the output of the model. We need to use `Enzyme.jl` to do this.

```julia
function loss_function(model, ps, st, x, y)
    pred, _ = model(x, ps, st)
    return MSELoss()(pred, y)
end
```

We will use `Zygote.jl` to compute the gradient of the loss function for the vanilla model.

```julia
loss_function(model, ps, st, x, y)

∂ps_zyg = only(Zygote.gradient(ps -> loss_function(model, ps, st, x, y), ps))
```

```ansi
(layer_1 = (weight = Float32[0.26016673 -0.09287719; -0.028029906 -0.013659926; … ; -0.073841624 -0.060032383; 0.042984407 0.051605422], bias = Float32[0.12923913, -0.0094058225, -0.026362807, -0.014524889, 0.013915392, 0.093436174, 0.08193636, 0.007762786, 0.0010442142, 0.018755753  …  0.06704105, -0.043209508, 0.10486872, 0.014353446, 0.02422882, -0.06582928, 0.010303016, 0.098782696, 0.067849405, -0.082684055]), layer_2 = (weight = Float32[-0.004457889 0.00021804233 … -0.0033274863 -0.008014374; 0.13051206 -0.0046890336 … 0.038353663 0.0933025; … ; -0.041253403 0.00088798517 … 0.0007470122 -0.020347351; 0.06339174 -0.002119769 … 0.0121456925 0.06877028], bias = Float32[-0.006240551, 0.13950492, -0.22439212, -0.11326962, -0.023160838, 0.14702772, 0.03519612, 0.13981938, -0.23715456, 0.32662553  …  -0.014224287, 0.009401775, 0.18295962, 0.13164552, 0.16955197, -0.110567965, -0.0074348953, 0.11886866, -0.026588853, 0.03181577]), layer_3 = (weight = Float32[-0.6772371 -0.19355826 … 0.092198014 -0.33821833; -0.29864177 -0.09485074 … 0.022576148 -0.17590505], bias = Float32[-1.1515996, -0.556467]))
```

Now we will compile the gradient function using `Reactant.@compile`.

```julia
function enzyme_gradient(model, ps, st, x, y)
    return Enzyme.gradient(Enzyme.Reverse, Const(loss_function), Const(model),
        ps, Const(st), Const(x), Const(y))[2]
end

enzyme_gradient_compiled = @compile enzyme_gradient(model, ps_ra, st_ra, x_ra, y_ra)

∂ps_enzyme = enzyme_gradient_compiled(model, ps_ra, st_ra, x_ra, y_ra)
```

```ansi
(layer_1 = (weight = Reactant.ConcretePJRTArray{Float32, 2, 1}(Float32[0.26016673 -0.09287716; -0.02802991 -0.013659924; … ; -0.07384164 -0.060032375; 0.042984392 0.05160541]), bias = Reactant.ConcretePJRTArray{Float32, 1, 1}(Float32[0.12923914, -0.009405823, -0.026362803, -0.014524889, 0.01391538, 0.093436226, 0.08193635, 0.007762787, 0.0010442168, 0.018755727  …  0.06704106, -0.043209508, 0.10486874, 0.01435344, 0.024228813, -0.06582926, 0.010303016, 0.098782696, 0.067849405, -0.08268405])), layer_2 = (weight = Reactant.ConcretePJRTArray{Float32, 2, 1}(Float32[-0.0044578887 0.00021804239 … -0.0033274866 -0.008014375; 0.13051206 -0.0046890336 … 0.03835366 0.093302496; … ; -0.041253407 0.0008879858 … 0.0007470099 -0.020347359; 0.06339174 -0.0021197703 … 0.0121456925 0.06877028]), bias = Reactant.ConcretePJRTArray{Float32, 1, 1}(Float32[-0.006240551, 0.13950492, -0.22439212, -0.11326965, -0.02316084, 0.14702772, 0.035196126, 0.1398194, -0.23715453, 0.32662556  …  -0.014224287, 0.00940178, 0.18295962, 0.13164552, 0.16955197, -0.110567965, -0.0074349036, 0.11886866, -0.026588857, 0.03181577])), layer_3 = (weight = Reactant.ConcretePJRTArray{Float32, 2, 1}(Float32[-0.6772371 -0.1935583 … 0.092198 -0.3382184; -0.29864174 -0.09485076 … 0.022576138 -0.17590503]), bias = Reactant.ConcretePJRTArray{Float32, 1, 1}(Float32[-1.1515996, -0.556467])))
```

Now we check the difference:

```julia
fmap(Broadcast.BroadcastFunction(-), ∂ps_zyg, ∂ps_enzyme |> cpu_device())
```

```ansi
(layer_1 = (weight = Float32[0.0 -2.9802322f-8; 3.7252903f-9 -1.8626451f-9; … ; 1.4901161f-8 -7.450581f-9; 1.4901161f-8 1.1175871f-8], bias = Float32[-1.4901161f-8, 9.313226f-10, -3.7252903f-9, 0.0, 1.1175871f-8, -5.2154064f-8, 7.450581f-9, -9.313226f-10, -2.561137f-9, 2.6077032f-8  …  -1.4901161f-8, 0.0, -2.2351742f-8, 5.5879354f-9, 7.450581f-9, -1.4901161f-8, 0.0, 0.0, 0.0, -7.450581f-9]), layer_2 = (weight = Float32[-4.656613f-10 -5.820766f-11 … 2.3283064f-10 9.313226f-10; 0.0 0.0 … 3.7252903f-9 7.450581f-9; … ; 3.7252903f-9 -6.4028427f-10 … 2.3283064f-9 7.450581f-9; 0.0 1.1641532f-9 … 0.0 0.0], bias = Float32[0.0, 0.0, 0.0, 2.9802322f-8, 1.8626451f-9, 0.0, -3.7252903f-9, -1.4901161f-8, -2.9802322f-8, -2.9802322f-8  …  0.0, -4.656613f-9, 0.0, 0.0, 0.0, 0.0, 8.381903f-9, 0.0, 3.7252903f-9, 0.0]), layer_3 = (weight = Float32[0.0 4.4703484f-8 … 1.4901161f-8 5.9604645f-8; -2.9802322f-8 2.2351742f-8 … 9.313226f-9 -1.4901161f-8], bias = Float32[0.0, 0.0]))
```

## Using the `TrainState` API {#compile\_lux\_model\_trainstate}

Now that we saw the low-level API let's see how to train the model without any of this boilerplate. Simply follow the following steps:

1. Create a device using `reactant_device`. Remember to load `Reactant.jl` before doing this.

2. Similar to other device functions move the model, parameters, states and data to the device. Note that you might want to use [`DeviceIterator`](/api/Accelerator_Support/MLDataDevices#MLDataDevices.DeviceIterator) to move the data loader to the device with an iterator.

3. Construct a `TrainState` using [`Training.TrainState`](/api/Lux/utilities#Lux.Training.TrainState).

4. And most importantly use `AutoEnzyme`/`AutoReactant` while calling [`Training.single_train_step!`](/api/Lux/utilities#Lux.Training.single_train_step!) or [`Training.single_train_step`](/api/Lux/utilities#Lux.Training.single_train_step).

```julia
model = Chain(
    Dense(2 => 4, gelu),
    Dense(4 => 4, gelu),
    Dense(4 => 2)
)
ps, st = Lux.setup(Random.default_rng(), model)

x_ra = [randn(Float32, 2, 32) for _ in 1:32]
y_ra = [xᵢ .^ 2 for xᵢ in x_ra]
ps_ra = ps |> xdev
st_ra = st |> xdev

dataloader = DeviceIterator(xdev, zip(x_ra, y_ra))

function train_model(model, ps, st, dataloader)
    train_state = Training.TrainState(model, ps, st, Adam(0.001f0))

    for iteration in 1:1000
        for (i, (xᵢ, yᵢ)) in enumerate(dataloader)
            _, loss, _, train_state = Training.single_train_step!(
                AutoEnzyme(), MSELoss(), (xᵢ, yᵢ), train_state
            )
            if (iteration % 100 == 0 || iteration == 1) && i == 1
                @printf("Iter: [%4d/%4d]\tLoss: %.8f\n", iteration, 1000, loss)
            end
        end
    end

    return train_state
end

train_model(model, ps_ra, st_ra, dataloader)
```

```ansi
Iter: [   1/1000]	Loss: 13.22820091
Iter: [ 100/1000]	Loss: 2.58897185
Iter: [ 200/1000]	Loss: 1.14364672
Iter: [ 300/1000]	Loss: 0.37711826
Iter: [ 400/1000]	Loss: 0.13414203
Iter: [ 500/1000]	Loss: 0.05696347
Iter: [ 600/1000]	Loss: 0.03033214
Iter: [ 700/1000]	Loss: 0.01917144
Iter: [ 800/1000]	Loss: 0.01335704
Iter: [ 900/1000]	Loss: 0.01001807
Iter: [1000/1000]	Loss: 0.00796757
```
