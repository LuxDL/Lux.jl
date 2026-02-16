---
url: /dev/manual/compiling_lux_models.md
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
(Float32[0.01586983 0.010564316 … -0.4137662 0.018748946; 0.078654006 0.06953075 … -0.2340262 0.21624328], (layer_1 = NamedTuple(), layer_2 = NamedTuple(), layer_3 = NamedTuple()))
```

To run it using `XLA` we need to compile the model. We can do this using the `Reactant.@compile` macro. Note that the inputs need to be moved to the device using [`reactant_device`](/api/Accelerator_Support/MLDataDevices#MLDataDevices.reactant_device) first.

```julia
model_compiled = @compile model(x_ra, ps_ra, Lux.testmode(st_ra))
```

```ansi
Reactant compiled function Chain{@NamedTuple{layer_1::Dense{typeof(gelu_tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Dense{typeof(gelu_tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_3::Dense{typeof(identity), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing}((layer_1 = Dense(2 => 32, gelu_tanh), layer_2 = Dense(32 => 32, gelu_tanh), layer_3 = Dense(32 => 2)), nothing) (with tag ##Chain{@NamedTuple{layer_1::Dense{typeof(gelu_tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Dense{typeof(gelu_tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_3::Dense{typeof(identity), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing}((layer_1 = Dense(2 => 32, gelu_tanh), layer_2 = Dense(32 => 32, gelu_tanh), layer_3 = Dense(32 => 2)), nothing)_reactant#960807)
```

Now we can test the difference between the results:

```julia
pred_compiled, _ = model_compiled(x_ra, ps_ra, Lux.testmode(st_ra))

pred_lux .- Array(pred_compiled)
```

```ansi
2×32 Matrix{Float32}:
 3.72529f-8  7.45058f-9  1.11759f-8  5.96046f-8  …  8.9407f-8    1.86265f-8
 7.45058f-9  2.23517f-8  0.0         2.23517f-7     1.49012f-8  -2.98023f-8
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
(layer_1 = (weight = Float32[0.26016673 -0.09287718; -0.02802991 -0.013659927; … ; -0.07384164 -0.06003239; 0.042984415 0.051605415], bias = Float32[0.12923914, -0.009405827, -0.0263628, -0.014524895, 0.013915386, 0.093436174, 0.08193636, 0.0077627874, 0.001044218, 0.018755747  …  0.067041054, -0.043209504, 0.10486872, 0.014353438, 0.024228808, -0.06582927, 0.010303013, 0.098782696, 0.06784941, -0.08268406]), layer_2 = (weight = Float32[-0.004457889 0.00021804236 … -0.003327486 -0.008014375; 0.13051204 -0.0046890336 … 0.038353663 0.093302496; … ; -0.041253403 0.000887985 … 0.000747012 -0.020347353; 0.06339173 -0.0021197682 … 0.012145687 0.06877027], bias = Float32[-0.0062405514, 0.13950492, -0.22439212, -0.113269635, -0.023160838, 0.14702773, 0.03519612, 0.13981938, -0.23715457, 0.32662556  …  -0.014224289, 0.009401775, 0.18295962, 0.13164552, 0.16955197, -0.110567965, -0.0074348953, 0.118868664, -0.026588853, 0.031815764]), layer_3 = (weight = Float32[-0.6772371 -0.19355826 … 0.092198014 -0.33821836; -0.29864177 -0.09485075 … 0.022576144 -0.17590499], bias = Float32[-1.1515998, -0.55646694]))
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
(layer_1 = (weight = Reactant.ConcretePJRTArray{Float32, 2, 1}(Float32[0.2601668 -0.09287718; -0.028029913 -0.013659928; … ; -0.07384164 -0.060032383; 0.042984392 0.051605426]), bias = Reactant.ConcretePJRTArray{Float32, 1, 1}(Float32[0.12923916, -0.009405825, -0.026362803, -0.0145248845, 0.0139153795, 0.09343623, 0.081936345, 0.007762788, 0.001044217, 0.018755728  …  0.06704106, -0.043209504, 0.104868725, 0.014353448, 0.024228806, -0.06582926, 0.01030301, 0.098782696, 0.067849405, -0.082684055])), layer_2 = (weight = Reactant.ConcretePJRTArray{Float32, 2, 1}(Float32[-0.004457889 0.00021804242 … -0.0033274868 -0.008014374; 0.13051206 -0.0046890336 … 0.03835366 0.093302496; … ; -0.041253403 0.00088798587 … 0.00074701075 -0.020347355; 0.063391745 -0.00211977 … 0.012145701 0.068770304]), bias = Reactant.ConcretePJRTArray{Float32, 1, 1}(Float32[-0.0062405514, 0.13950491, -0.22439212, -0.11326962, -0.02316084, 0.14702773, 0.035196126, 0.1398194, -0.23715451, 0.32662556  …  -0.01422429, 0.009401777, 0.18295962, 0.13164552, 0.16955195, -0.11056796, -0.0074349036, 0.11886866, -0.026588855, 0.031815786])), layer_3 = (weight = Reactant.ConcretePJRTArray{Float32, 2, 1}(Float32[-0.6772371 -0.19355828 … 0.092198 -0.33821842; -0.29864174 -0.09485076 … 0.022576137 -0.1759051]), bias = Reactant.ConcretePJRTArray{Float32, 1, 1}(Float32[-1.1515996, -0.55646694])))
```

Now we check the difference:

```julia
fmap(Broadcast.BroadcastFunction(-), ∂ps_zyg, ∂ps_enzyme |> cpu_device())
```

```ansi
(layer_1 = (weight = Float32[-5.9604645f-8 0.0; 3.7252903f-9 9.313226f-10; … ; 0.0 -7.450581f-9; 2.2351742f-8 -1.1175871f-8], bias = Float32[-1.4901161f-8, -1.8626451f-9, 3.7252903f-9, -1.0244548f-8, 6.519258f-9, -5.9604645f-8, 1.4901161f-8, -4.656613f-10, 9.313226f-10, 1.8626451f-8  …  -7.450581f-9, 0.0, -7.450581f-9, -9.313226f-9, 1.8626451f-9, -7.450581f-9, 2.7939677f-9, 0.0, 7.450581f-9, -7.450581f-9]), layer_2 = (weight = Float32[0.0 -5.820766f-11 … 6.9849193f-10 -9.313226f-10; -1.4901161f-8 0.0 … 3.7252903f-9 0.0; … ; 0.0 -8.731149f-10 … 1.2223609f-9 1.8626451f-9; -1.4901161f-8 1.8626451f-9 … -1.3969839f-8 -3.7252903f-8], bias = Float32[0.0, 1.4901161f-8, 0.0, -1.4901161f-8, 1.8626451f-9, 0.0, -3.7252903f-9, -1.4901161f-8, -5.9604645f-8, 0.0  …  9.313226f-10, -1.8626451f-9, 0.0, 0.0, 1.4901161f-8, -7.450581f-9, 8.381903f-9, 7.450581f-9, 1.8626451f-9, -2.2351742f-8]), layer_3 = (weight = Float32[0.0 1.4901161f-8 … 1.4901161f-8 5.9604645f-8; -2.9802322f-8 1.4901161f-8 … 7.450581f-9 1.0430813f-7], bias = Float32[-1.1920929f-7, 0.0]))
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
Iter: [ 100/1000]	Loss: 2.58897161
Iter: [ 200/1000]	Loss: 1.14364624
Iter: [ 300/1000]	Loss: 0.37711838
Iter: [ 400/1000]	Loss: 0.13413867
Iter: [ 500/1000]	Loss: 0.05696292
Iter: [ 600/1000]	Loss: 0.03033197
Iter: [ 700/1000]	Loss: 0.01917133
Iter: [ 800/1000]	Loss: 0.01335695
Iter: [ 900/1000]	Loss: 0.01001796
Iter: [1000/1000]	Loss: 0.00796750
```
