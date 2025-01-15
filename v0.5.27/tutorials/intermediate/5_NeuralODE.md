


<a id='MNIST-Classification-using-Neural-ODEs'></a>

# MNIST Classification using Neural ODEs


To understand Neural ODEs, users should look up [these lecture notes](https://book.sciml.ai/notes/11-Differentiable_Programming_and_Neural_Differential_Equations/). We recommend users to directly use [DiffEqFlux.jl](https://docs.sciml.ai/DiffEqFlux/stable/), instead of implementing Neural ODEs from scratch.


<a id='Package-Imports'></a>

## Package Imports


```julia
using Lux, ComponentArrays, SciMLSensitivity, LuxAMDGPU, LuxCUDA, Optimisers,
      OrdinaryDiffEq, Random, Statistics, Zygote, OneHotArrays, InteractiveUtils, Printf
import MLDatasets: MNIST
import MLUtils: DataLoader, splitobs

CUDA.allowscalar(false)
```


<a id='Loading-MNIST'></a>

## Loading MNIST


```julia
function loadmnist(batchsize, train_split)
    # Load MNIST: Only 1500 for demonstration purposes
    N = 1500
    dataset = MNIST(; split=:train)
    imgs = dataset.features[:, :, 1:N]
    labels_raw = dataset.targets[1:N]

    # Process images into (H,W,C,BS) batches
    x_data = Float32.(reshape(imgs, size(imgs, 1), size(imgs, 2), 1, size(imgs, 3)))
    y_data = onehotbatch(labels_raw, 0:9)
    (x_train, y_train), (x_test, y_test) = splitobs((x_data, y_data); at=train_split)

    return (
        # Use DataLoader to automatically minibatch and shuffle the data
        DataLoader(collect.((x_train, y_train)); batchsize, shuffle=true),
        # Don't shuffle the test data
        DataLoader(collect.((x_test, y_test)); batchsize, shuffle=false))
end
```


```
loadmnist (generic function with 1 method)
```


<a id='Define-the-Neural-ODE-Layer'></a>

## Define the Neural ODE Layer


The NeuralODE is a ContainerLayer, which stores a `model`. The parameters and states of the NeuralODE are same as those of the underlying model.


```julia
struct NeuralODE{M <: Lux.AbstractExplicitLayer, So, T, K} <:
       Lux.AbstractExplicitContainerLayer{(:model,)}
    model::M
    solver::So
    tspan::T
    kwargs::K
end

function NeuralODE(
        model::Lux.AbstractExplicitLayer; solver=Tsit5(), tspan=(0.0f0, 1.0f0), kwargs...)
    return NeuralODE(model, solver, tspan, kwargs)
end
```


```
Main.var"##225".NeuralODE
```


OrdinaryDiffEq.jl can deal with non-Vector Inputs! However, certain discrete sensitivities like `ReverseDiffAdjoint` can't handle non-Vector inputs. Hence, we need to convert the input and output of the ODE solver to a Vector.


```julia
function (n::NeuralODE)(x, ps, st)
    function dudt(u, p, t)
        u_, st = n.model(reshape(u, size(x)), p, st)
        return vec(u_)
    end
    prob = ODEProblem{false}(ODEFunction{false}(dudt), vec(x), n.tspan, ps)
    return solve(prob, n.solver; n.kwargs...), st
end

@views diffeqsol_to_array(l::Int, x::ODESolution) = reshape(last(x.u), (l, :))
@views diffeqsol_to_array(l::Int, x::AbstractMatrix) = reshape(x[:, end], (l, :))
```


```
diffeqsol_to_array (generic function with 2 methods)
```


<a id='Create-and-Initialize-the-Neural-ODE-Layer'></a>

## Create and Initialize the Neural ODE Layer


```julia
function create_model(model_fn=NeuralODE; dev=gpu_device(), use_named_tuple::Bool=false,
        sensealg=InterpolatingAdjoint(; autojacvec=ZygoteVJP()))
    # Construct the Neural ODE Model
    model = Chain(FlattenLayer(),
        Dense(784 => 20, tanh),
        model_fn(Chain(Dense(20 => 10, tanh), Dense(10 => 10, tanh), Dense(10 => 20, tanh));
            save_everystep=false, reltol=1.0f-3,
            abstol=1.0f-3, save_start=false, sensealg),
        Base.Fix1(diffeqsol_to_array, 20),
        Dense(20 => 10))

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    ps, st = Lux.setup(rng, model)
    ps = (use_named_tuple ? ps : ComponentArray(ps)) |> dev
    st = st |> dev

    return model, ps, st
end
```


```
create_model (generic function with 2 methods)
```


<a id='Define-Utility-Functions'></a>

## Define Utility Functions


```julia
logitcrossentropy(y_pred, y) = mean(-sum(y .* logsoftmax(y_pred); dims=1))

function loss(x, y, model, ps, st)
    y_pred, st = model(x, ps, st)
    return logitcrossentropy(y_pred, y), st
end

function accuracy(model, ps, st, dataloader; dev=gpu_device())
    total_correct, total = 0, 0
    st = Lux.testmode(st)
    cpu_dev = cpu_device()
    for (x, y) in dataloader
        target_class = onecold(y)
        predicted_class = onecold(cpu_dev(first(model(dev(x), ps, st))))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end
```


```
accuracy (generic function with 1 method)
```


<a id='Training'></a>

## Training


```julia
function train(model_function; cpu::Bool=false, kwargs...)
    dev = cpu ? cpu_device() : gpu_device()
    model, ps, st = create_model(model_function; dev, kwargs...)

    # Training
    train_dataloader, test_dataloader = loadmnist(128, 0.9)

    opt = Adam(0.001f0)
    st_opt = Optimisers.setup(opt, ps)

    ### Warmup the Model
    img = dev(train_dataloader.data[1][:, :, :, 1:1])
    lab = dev(train_dataloader.data[2][:, 1:1])
    loss(img, lab, model, ps, st)
    (l, _), back = pullback(p -> loss(img, lab, model, p, st), ps)
    back((one(l), nothing))

    ### Lets train the model
    nepochs = 9
    for epoch in 1:nepochs
        stime = time()
        for (x, y) in train_dataloader
            x = dev(x)
            y = dev(y)
            (l, st), back = pullback(p -> loss(x, y, model, p, st), ps)
            ### We need to add `nothing`s equal to the number of returned values - 1
            gs = back((one(l), nothing))[1]
            st_opt, ps = Optimisers.update(st_opt, ps, gs)
        end
        ttime = time() - stime

        tr_acc = accuracy(model, ps, st, train_dataloader; dev)
        te_acc = accuracy(model, ps, st, test_dataloader; dev)
        @printf "[%d/%d] \t Time %.2fs \t Training Accuracy: %.5f%% \t Test Accuracy: %.5f%%\n" epoch nepochs ttime tr_acc te_acc
    end
end

train(NeuralODE)
```


```
[1/9] 	 Time 3.76s 	 Training Accuracy: 0.50741% 	 Test Accuracy: 0.45333%
[2/9] 	 Time 0.57s 	 Training Accuracy: 0.70741% 	 Test Accuracy: 0.66667%
[3/9] 	 Time 0.32s 	 Training Accuracy: 0.77852% 	 Test Accuracy: 0.71333%
[4/9] 	 Time 0.33s 	 Training Accuracy: 0.81037% 	 Test Accuracy: 0.75333%
[5/9] 	 Time 0.66s 	 Training Accuracy: 0.82667% 	 Test Accuracy: 0.78000%
[6/9] 	 Time 0.31s 	 Training Accuracy: 0.84148% 	 Test Accuracy: 0.78667%
[7/9] 	 Time 0.30s 	 Training Accuracy: 0.85481% 	 Test Accuracy: 0.80667%
[8/9] 	 Time 0.30s 	 Training Accuracy: 0.86815% 	 Test Accuracy: 0.82000%
[9/9] 	 Time 0.57s 	 Training Accuracy: 0.87407% 	 Test Accuracy: 0.84000%

```


We can also change the sensealg and train the model! `GaussAdjoint` allows you to use any arbitrary parameter structure and not just a flat vector (`ComponentArray`).


```julia
train(NeuralODE; sensealg=GaussAdjoint(; autojacvec=ZygoteVJP()), use_named_tuple=true)
```


```
[1/9] 	 Time 2.68s 	 Training Accuracy: 0.49630% 	 Test Accuracy: 0.38000%
[2/9] 	 Time 0.27s 	 Training Accuracy: 0.70593% 	 Test Accuracy: 0.65333%
[3/9] 	 Time 0.28s 	 Training Accuracy: 0.78296% 	 Test Accuracy: 0.72000%
[4/9] 	 Time 0.28s 	 Training Accuracy: 0.80889% 	 Test Accuracy: 0.74000%
[5/9] 	 Time 0.28s 	 Training Accuracy: 0.82370% 	 Test Accuracy: 0.76667%
[6/9] 	 Time 0.29s 	 Training Accuracy: 0.84074% 	 Test Accuracy: 0.78667%
[7/9] 	 Time 0.30s 	 Training Accuracy: 0.85630% 	 Test Accuracy: 0.81333%
[8/9] 	 Time 0.31s 	 Training Accuracy: 0.86370% 	 Test Accuracy: 0.82000%
[9/9] 	 Time 0.33s 	 Training Accuracy: 0.87704% 	 Test Accuracy: 0.82667%

```


But remember some AD backends like `ReverseDiff` is not GPU compatible. For a model this size, you will notice that training time is significantly lower for training on CPU than on GPU.


```julia
train(NeuralODE; sensealg=InterpolatingAdjoint(; autojacvec=ReverseDiffVJP()), cpu=true)
```


```
[1/9] 	 Time 1.10s 	 Training Accuracy: 0.50963% 	 Test Accuracy: 0.43333%
[2/9] 	 Time 0.26s 	 Training Accuracy: 0.69630% 	 Test Accuracy: 0.66000%
[3/9] 	 Time 0.23s 	 Training Accuracy: 0.77926% 	 Test Accuracy: 0.71333%
[4/9] 	 Time 0.44s 	 Training Accuracy: 0.80741% 	 Test Accuracy: 0.76667%
[5/9] 	 Time 0.25s 	 Training Accuracy: 0.82519% 	 Test Accuracy: 0.78000%
[6/9] 	 Time 0.25s 	 Training Accuracy: 0.84074% 	 Test Accuracy: 0.78667%
[7/9] 	 Time 0.24s 	 Training Accuracy: 0.85333% 	 Test Accuracy: 0.80667%
[8/9] 	 Time 0.32s 	 Training Accuracy: 0.86593% 	 Test Accuracy: 0.81333%
[9/9] 	 Time 0.23s 	 Training Accuracy: 0.87704% 	 Test Accuracy: 0.82000%

```


For completeness, let's also test out discrete sensitivities!


```julia
train(NeuralODE; sensealg=ReverseDiffAdjoint(), cpu=true)
```


```
[1/9] 	 Time 7.93s 	 Training Accuracy: 0.50963% 	 Test Accuracy: 0.43333%
[2/9] 	 Time 7.72s 	 Training Accuracy: 0.69630% 	 Test Accuracy: 0.66000%
[3/9] 	 Time 7.02s 	 Training Accuracy: 0.77926% 	 Test Accuracy: 0.71333%
[4/9] 	 Time 7.43s 	 Training Accuracy: 0.80741% 	 Test Accuracy: 0.76667%
[5/9] 	 Time 9.39s 	 Training Accuracy: 0.82519% 	 Test Accuracy: 0.78000%
[6/9] 	 Time 9.35s 	 Training Accuracy: 0.84074% 	 Test Accuracy: 0.78667%
[7/9] 	 Time 9.89s 	 Training Accuracy: 0.85333% 	 Test Accuracy: 0.80667%
[8/9] 	 Time 10.36s 	 Training Accuracy: 0.86593% 	 Test Accuracy: 0.81333%
[9/9] 	 Time 9.94s 	 Training Accuracy: 0.87704% 	 Test Accuracy: 0.82000%

```


<a id='Alternate-Implementation-using-Stateful-Layer'></a>

## Alternate Implementation using Stateful Layer


Starting `v0.5.5`, Lux provides a `Lux.Experimental.StatefulLuxLayer` which can be used to avoid the [`Box`ing of `st`](https://github.com/JuliaLang/julia/issues/15276).


```julia
struct StatefulNeuralODE{M <: Lux.AbstractExplicitLayer, So, T, K} <:
       Lux.AbstractExplicitContainerLayer{(:model,)}
    model::M
    solver::So
    tspan::T
    kwargs::K
end

function StatefulNeuralODE(
        model::Lux.AbstractExplicitLayer; solver=Tsit5(), tspan=(0.0f0, 1.0f0), kwargs...)
    return StatefulNeuralODE(model, solver, tspan, kwargs)
end

function (n::StatefulNeuralODE)(x, ps, st)
    st_model = Lux.StatefulLuxLayer(n.model, ps, st)
    dudt(u, p, t) = st_model(u, p)
    prob = ODEProblem{false}(ODEFunction{false}(dudt), x, n.tspan, ps)
    return solve(prob, n.solver; n.kwargs...), st_model.st
end
```


<a id='Train-the-new-Stateful-Neural-ODE'></a>

## Train the new Stateful Neural ODE


```julia
train(StatefulNeuralODE)
```


```
[1/9] 	 Time 1.34s 	 Training Accuracy: 0.49852% 	 Test Accuracy: 0.40667%
[2/9] 	 Time 0.71s 	 Training Accuracy: 0.70296% 	 Test Accuracy: 0.66667%
[3/9] 	 Time 0.29s 	 Training Accuracy: 0.78074% 	 Test Accuracy: 0.71333%
[4/9] 	 Time 0.33s 	 Training Accuracy: 0.80741% 	 Test Accuracy: 0.76000%
[5/9] 	 Time 0.43s 	 Training Accuracy: 0.82000% 	 Test Accuracy: 0.78000%
[6/9] 	 Time 0.41s 	 Training Accuracy: 0.84444% 	 Test Accuracy: 0.79333%
[7/9] 	 Time 0.42s 	 Training Accuracy: 0.85704% 	 Test Accuracy: 0.82000%
[8/9] 	 Time 0.36s 	 Training Accuracy: 0.87037% 	 Test Accuracy: 0.80667%
[9/9] 	 Time 0.91s 	 Training Accuracy: 0.88000% 	 Test Accuracy: 0.82667%

```


We might not see a significant difference in the training time, but let us investigate the type stabilities of the layers.


<a id='Type-Stability'></a>

## Type Stability


```julia
model, ps, st = create_model(NeuralODE)

model_stateful, ps_stateful, st_stateful = create_model(StatefulNeuralODE)

x = gpu_device()(ones(Float32, 28, 28, 1, 3));
```


NeuralODE is not type stable due to the boxing of `st`


```julia
@code_warntype model(x, ps, st)
```


```
MethodInstance for (::Lux.Chain{@NamedTuple{layer_1::Lux.FlattenLayer{Nothing}, layer_2::Lux.Dense{true, typeof(NNlib.tanh_fast), typeof(WeightInitializers.glorot_uniform), typeof(WeightInitializers.zeros32)}, layer_3::Main.var"##225".NeuralODE{Lux.Chain{@NamedTuple{layer_1::Lux.Dense{true, typeof(NNlib.tanh_fast), typeof(WeightInitializers.glorot_uniform), typeof(WeightInitializers.zeros32)}, layer_2::Lux.Dense{true, typeof(NNlib.tanh_fast), typeof(WeightInitializers.glorot_uniform), typeof(WeightInitializers.zeros32)}, layer_3::Lux.Dense{true, typeof(NNlib.tanh_fast), typeof(WeightInitializers.glorot_uniform), typeof(WeightInitializers.zeros32)}}, Nothing}, OrdinaryDiffEq.Tsit5{typeof(OrdinaryDiffEq.trivial_limiter!), typeof(OrdinaryDiffEq.trivial_limiter!), Static.False}, Tuple{Float32, Float32}, Base.Pairs{Symbol, Any, NTuple{5, Symbol}, @NamedTuple{save_everystep::Bool, reltol::Float32, abstol::Float32, save_start::Bool, sensealg::SciMLSensitivity.InterpolatingAdjoint{0, true, Val{:central}, SciMLSensitivity.ZygoteVJP}}}}, layer_4::Lux.WrappedFunction{Base.Fix1{typeof(Main.var"##225".diffeqsol_to_array), Int64}}, layer_5::Lux.Dense{true, typeof(identity), typeof(WeightInitializers.glorot_uniform), typeof(WeightInitializers.zeros32)}}, Nothing})(::CUDA.CuArray{Float32, 4, CUDA.Mem.DeviceBuffer}, ::ComponentArrays.ComponentVector{Float32, CUDA.CuArray{Float32, 1, CUDA.Mem.DeviceBuffer}, Tuple{ComponentArrays.Axis{(layer_1 = 1:0, layer_2 = ViewAxis(1:15700, Axis(weight = ViewAxis(1:15680, ShapedAxis((20, 784))), bias = ViewAxis(15681:15700, ShapedAxis((20, 1))))), layer_3 = ViewAxis(15701:16240, Axis(layer_1 = ViewAxis(1:210, Axis(weight = ViewAxis(1:200, ShapedAxis((10, 20))), bias = ViewAxis(201:210, ShapedAxis((10, 1))))), layer_2 = ViewAxis(211:320, Axis(weight = ViewAxis(1:100, ShapedAxis((10, 10))), bias = ViewAxis(101:110, ShapedAxis((10, 1))))), layer_3 = ViewAxis(321:540, Axis(weight = ViewAxis(1:200, ShapedAxis((20, 10))), bias = ViewAxis(201:220, ShapedAxis((20, 1))))))), layer_4 = 16241:16240, layer_5 = ViewAxis(16241:16450, Axis(weight = ViewAxis(1:200, ShapedAxis((10, 20))), bias = ViewAxis(201:210, ShapedAxis((10, 1))))))}}}, ::@NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}, layer_3::@NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}, layer_3::@NamedTuple{}}, layer_4::@NamedTuple{}, layer_5::@NamedTuple{}})
  from (c::Lux.Chain)(x, ps, st::NamedTuple) @ Lux /var/lib/buildkite-agent/builds/gpuci-11/julialang/lux-dot-jl/src/layers/containers.jl:479
Arguments
  c::Lux.Chain{@NamedTuple{layer_1::Lux.FlattenLayer{Nothing}, layer_2::Lux.Dense{true, typeof(NNlib.tanh_fast), typeof(WeightInitializers.glorot_uniform), typeof(WeightInitializers.zeros32)}, layer_3::Main.var"##225".NeuralODE{Lux.Chain{@NamedTuple{layer_1::Lux.Dense{true, typeof(NNlib.tanh_fast), typeof(WeightInitializers.glorot_uniform), typeof(WeightInitializers.zeros32)}, layer_2::Lux.Dense{true, typeof(NNlib.tanh_fast), typeof(WeightInitializers.glorot_uniform), typeof(WeightInitializers.zeros32)}, layer_3::Lux.Dense{true, typeof(NNlib.tanh_fast), typeof(WeightInitializers.glorot_uniform), typeof(WeightInitializers.zeros32)}}, Nothing}, OrdinaryDiffEq.Tsit5{typeof(OrdinaryDiffEq.trivial_limiter!), typeof(OrdinaryDiffEq.trivial_limiter!), Static.False}, Tuple{Float32, Float32}, Base.Pairs{Symbol, Any, NTuple{5, Symbol}, @NamedTuple{save_everystep::Bool, reltol::Float32, abstol::Float32, save_start::Bool, sensealg::SciMLSensitivity.InterpolatingAdjoint{0, true, Val{:central}, SciMLSensitivity.ZygoteVJP}}}}, layer_4::Lux.WrappedFunction{Base.Fix1{typeof(Main.var"##225".diffeqsol_to_array), Int64}}, layer_5::Lux.Dense{true, typeof(identity), typeof(WeightInitializers.glorot_uniform), typeof(WeightInitializers.zeros32)}}, Nothing}
  x::CUDA.CuArray{Float32, 4, CUDA.Mem.DeviceBuffer}
  ps::ComponentArrays.ComponentVector{Float32, CUDA.CuArray{Float32, 1, CUDA.Mem.DeviceBuffer}, Tuple{ComponentArrays.Axis{(layer_1 = 1:0, layer_2 = ViewAxis(1:15700, Axis(weight = ViewAxis(1:15680, ShapedAxis((20, 784))), bias = ViewAxis(15681:15700, ShapedAxis((20, 1))))), layer_3 = ViewAxis(15701:16240, Axis(layer_1 = ViewAxis(1:210, Axis(weight = ViewAxis(1:200, ShapedAxis((10, 20))), bias = ViewAxis(201:210, ShapedAxis((10, 1))))), layer_2 = ViewAxis(211:320, Axis(weight = ViewAxis(1:100, ShapedAxis((10, 10))), bias = ViewAxis(101:110, ShapedAxis((10, 1))))), layer_3 = ViewAxis(321:540, Axis(weight = ViewAxis(1:200, ShapedAxis((20, 10))), bias = ViewAxis(201:220, ShapedAxis((20, 1))))))), layer_4 = 16241:16240, layer_5 = ViewAxis(16241:16450, Axis(weight = ViewAxis(1:200, ShapedAxis((10, 20))), bias = ViewAxis(201:210, ShapedAxis((10, 1))))))}}}
  st::Core.Const((layer_1 = NamedTuple(), layer_2 = NamedTuple(), layer_3 = (layer_1 = NamedTuple(), layer_2 = NamedTuple(), layer_3 = NamedTuple()), layer_4 = NamedTuple(), layer_5 = NamedTuple()))
Body::TUPLE{CUDA.CUARRAY{FLOAT32, 2, CUDA.MEM.DEVICEBUFFER}, NAMEDTUPLE{(:LAYER_1, :LAYER_2, :LAYER_3, :LAYER_4, :LAYER_5), <:TUPLE{@NAMEDTUPLE{}, @NAMEDTUPLE{}, ANY, @NAMEDTUPLE{}, @NAMEDTUPLE{}}}}
1 ─ %1 = Base.getproperty(c, :layers)::@NamedTuple{layer_1::Lux.FlattenLayer{Nothing}, layer_2::Lux.Dense{true, typeof(NNlib.tanh_fast), typeof(WeightInitializers.glorot_uniform), typeof(WeightInitializers.zeros32)}, layer_3::Main.var"##225".NeuralODE{Lux.Chain{@NamedTuple{layer_1::Lux.Dense{true, typeof(NNlib.tanh_fast), typeof(WeightInitializers.glorot_uniform), typeof(WeightInitializers.zeros32)}, layer_2::Lux.Dense{true, typeof(NNlib.tanh_fast), typeof(WeightInitializers.glorot_uniform), typeof(WeightInitializers.zeros32)}, layer_3::Lux.Dense{true, typeof(NNlib.tanh_fast), typeof(WeightInitializers.glorot_uniform), typeof(WeightInitializers.zeros32)}}, Nothing}, OrdinaryDiffEq.Tsit5{typeof(OrdinaryDiffEq.trivial_limiter!), typeof(OrdinaryDiffEq.trivial_limiter!), Static.False}, Tuple{Float32, Float32}, Base.Pairs{Symbol, Any, NTuple{5, Symbol}, @NamedTuple{save_everystep::Bool, reltol::Float32, abstol::Float32, save_start::Bool, sensealg::SciMLSensitivity.InterpolatingAdjoint{0, true, Val{:central}, SciMLSensitivity.ZygoteVJP}}}}, layer_4::Lux.WrappedFunction{Base.Fix1{typeof(Main.var"##225".diffeqsol_to_array), Int64}}, layer_5::Lux.Dense{true, typeof(identity), typeof(WeightInitializers.glorot_uniform), typeof(WeightInitializers.zeros32)}}
│   %2 = Lux.applychain(%1, x, ps, st)::TUPLE{CUDA.CUARRAY{FLOAT32, 2, CUDA.MEM.DEVICEBUFFER}, NAMEDTUPLE{(:LAYER_1, :LAYER_2, :LAYER_3, :LAYER_4, :LAYER_5), <:TUPLE{@NAMEDTUPLE{}, @NAMEDTUPLE{}, ANY, @NAMEDTUPLE{}, @NAMEDTUPLE{}}}}
└──      return %2


```


We avoid the problem entirely by using `StatefulNeuralODE`


```julia
@code_warntype model_stateful(x, ps_stateful, st_stateful)
```


```
MethodInstance for (::Lux.Chain{@NamedTuple{layer_1::Lux.FlattenLayer{Nothing}, layer_2::Lux.Dense{true, typeof(NNlib.tanh_fast), typeof(WeightInitializers.glorot_uniform), typeof(WeightInitializers.zeros32)}, layer_3::Main.var"##225".StatefulNeuralODE{Lux.Chain{@NamedTuple{layer_1::Lux.Dense{true, typeof(NNlib.tanh_fast), typeof(WeightInitializers.glorot_uniform), typeof(WeightInitializers.zeros32)}, layer_2::Lux.Dense{true, typeof(NNlib.tanh_fast), typeof(WeightInitializers.glorot_uniform), typeof(WeightInitializers.zeros32)}, layer_3::Lux.Dense{true, typeof(NNlib.tanh_fast), typeof(WeightInitializers.glorot_uniform), typeof(WeightInitializers.zeros32)}}, Nothing}, OrdinaryDiffEq.Tsit5{typeof(OrdinaryDiffEq.trivial_limiter!), typeof(OrdinaryDiffEq.trivial_limiter!), Static.False}, Tuple{Float32, Float32}, Base.Pairs{Symbol, Any, NTuple{5, Symbol}, @NamedTuple{save_everystep::Bool, reltol::Float32, abstol::Float32, save_start::Bool, sensealg::SciMLSensitivity.InterpolatingAdjoint{0, true, Val{:central}, SciMLSensitivity.ZygoteVJP}}}}, layer_4::Lux.WrappedFunction{Base.Fix1{typeof(Main.var"##225".diffeqsol_to_array), Int64}}, layer_5::Lux.Dense{true, typeof(identity), typeof(WeightInitializers.glorot_uniform), typeof(WeightInitializers.zeros32)}}, Nothing})(::CUDA.CuArray{Float32, 4, CUDA.Mem.DeviceBuffer}, ::ComponentArrays.ComponentVector{Float32, CUDA.CuArray{Float32, 1, CUDA.Mem.DeviceBuffer}, Tuple{ComponentArrays.Axis{(layer_1 = 1:0, layer_2 = ViewAxis(1:15700, Axis(weight = ViewAxis(1:15680, ShapedAxis((20, 784))), bias = ViewAxis(15681:15700, ShapedAxis((20, 1))))), layer_3 = ViewAxis(15701:16240, Axis(layer_1 = ViewAxis(1:210, Axis(weight = ViewAxis(1:200, ShapedAxis((10, 20))), bias = ViewAxis(201:210, ShapedAxis((10, 1))))), layer_2 = ViewAxis(211:320, Axis(weight = ViewAxis(1:100, ShapedAxis((10, 10))), bias = ViewAxis(101:110, ShapedAxis((10, 1))))), layer_3 = ViewAxis(321:540, Axis(weight = ViewAxis(1:200, ShapedAxis((20, 10))), bias = ViewAxis(201:220, ShapedAxis((20, 1))))))), layer_4 = 16241:16240, layer_5 = ViewAxis(16241:16450, Axis(weight = ViewAxis(1:200, ShapedAxis((10, 20))), bias = ViewAxis(201:210, ShapedAxis((10, 1))))))}}}, ::@NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}, layer_3::@NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}, layer_3::@NamedTuple{}}, layer_4::@NamedTuple{}, layer_5::@NamedTuple{}})
  from (c::Lux.Chain)(x, ps, st::NamedTuple) @ Lux /var/lib/buildkite-agent/builds/gpuci-11/julialang/lux-dot-jl/src/layers/containers.jl:479
Arguments
  c::Lux.Chain{@NamedTuple{layer_1::Lux.FlattenLayer{Nothing}, layer_2::Lux.Dense{true, typeof(NNlib.tanh_fast), typeof(WeightInitializers.glorot_uniform), typeof(WeightInitializers.zeros32)}, layer_3::Main.var"##225".StatefulNeuralODE{Lux.Chain{@NamedTuple{layer_1::Lux.Dense{true, typeof(NNlib.tanh_fast), typeof(WeightInitializers.glorot_uniform), typeof(WeightInitializers.zeros32)}, layer_2::Lux.Dense{true, typeof(NNlib.tanh_fast), typeof(WeightInitializers.glorot_uniform), typeof(WeightInitializers.zeros32)}, layer_3::Lux.Dense{true, typeof(NNlib.tanh_fast), typeof(WeightInitializers.glorot_uniform), typeof(WeightInitializers.zeros32)}}, Nothing}, OrdinaryDiffEq.Tsit5{typeof(OrdinaryDiffEq.trivial_limiter!), typeof(OrdinaryDiffEq.trivial_limiter!), Static.False}, Tuple{Float32, Float32}, Base.Pairs{Symbol, Any, NTuple{5, Symbol}, @NamedTuple{save_everystep::Bool, reltol::Float32, abstol::Float32, save_start::Bool, sensealg::SciMLSensitivity.InterpolatingAdjoint{0, true, Val{:central}, SciMLSensitivity.ZygoteVJP}}}}, layer_4::Lux.WrappedFunction{Base.Fix1{typeof(Main.var"##225".diffeqsol_to_array), Int64}}, layer_5::Lux.Dense{true, typeof(identity), typeof(WeightInitializers.glorot_uniform), typeof(WeightInitializers.zeros32)}}, Nothing}
  x::CUDA.CuArray{Float32, 4, CUDA.Mem.DeviceBuffer}
  ps::ComponentArrays.ComponentVector{Float32, CUDA.CuArray{Float32, 1, CUDA.Mem.DeviceBuffer}, Tuple{ComponentArrays.Axis{(layer_1 = 1:0, layer_2 = ViewAxis(1:15700, Axis(weight = ViewAxis(1:15680, ShapedAxis((20, 784))), bias = ViewAxis(15681:15700, ShapedAxis((20, 1))))), layer_3 = ViewAxis(15701:16240, Axis(layer_1 = ViewAxis(1:210, Axis(weight = ViewAxis(1:200, ShapedAxis((10, 20))), bias = ViewAxis(201:210, ShapedAxis((10, 1))))), layer_2 = ViewAxis(211:320, Axis(weight = ViewAxis(1:100, ShapedAxis((10, 10))), bias = ViewAxis(101:110, ShapedAxis((10, 1))))), layer_3 = ViewAxis(321:540, Axis(weight = ViewAxis(1:200, ShapedAxis((20, 10))), bias = ViewAxis(201:220, ShapedAxis((20, 1))))))), layer_4 = 16241:16240, layer_5 = ViewAxis(16241:16450, Axis(weight = ViewAxis(1:200, ShapedAxis((10, 20))), bias = ViewAxis(201:210, ShapedAxis((10, 1))))))}}}
  st::Core.Const((layer_1 = NamedTuple(), layer_2 = NamedTuple(), layer_3 = (layer_1 = NamedTuple(), layer_2 = NamedTuple(), layer_3 = NamedTuple()), layer_4 = NamedTuple(), layer_5 = NamedTuple()))
Body::Tuple{CUDA.CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}, @NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}, layer_3::@NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}, layer_3::@NamedTuple{}}, layer_4::@NamedTuple{}, layer_5::@NamedTuple{}}}
1 ─ %1 = Base.getproperty(c, :layers)::@NamedTuple{layer_1::Lux.FlattenLayer{Nothing}, layer_2::Lux.Dense{true, typeof(NNlib.tanh_fast), typeof(WeightInitializers.glorot_uniform), typeof(WeightInitializers.zeros32)}, layer_3::Main.var"##225".StatefulNeuralODE{Lux.Chain{@NamedTuple{layer_1::Lux.Dense{true, typeof(NNlib.tanh_fast), typeof(WeightInitializers.glorot_uniform), typeof(WeightInitializers.zeros32)}, layer_2::Lux.Dense{true, typeof(NNlib.tanh_fast), typeof(WeightInitializers.glorot_uniform), typeof(WeightInitializers.zeros32)}, layer_3::Lux.Dense{true, typeof(NNlib.tanh_fast), typeof(WeightInitializers.glorot_uniform), typeof(WeightInitializers.zeros32)}}, Nothing}, OrdinaryDiffEq.Tsit5{typeof(OrdinaryDiffEq.trivial_limiter!), typeof(OrdinaryDiffEq.trivial_limiter!), Static.False}, Tuple{Float32, Float32}, Base.Pairs{Symbol, Any, NTuple{5, Symbol}, @NamedTuple{save_everystep::Bool, reltol::Float32, abstol::Float32, save_start::Bool, sensealg::SciMLSensitivity.InterpolatingAdjoint{0, true, Val{:central}, SciMLSensitivity.ZygoteVJP}}}}, layer_4::Lux.WrappedFunction{Base.Fix1{typeof(Main.var"##225".diffeqsol_to_array), Int64}}, layer_5::Lux.Dense{true, typeof(identity), typeof(WeightInitializers.glorot_uniform), typeof(WeightInitializers.zeros32)}}
│   %2 = Lux.applychain(%1, x, ps, st)::Tuple{CUDA.CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}, @NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}, layer_3::@NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}, layer_3::@NamedTuple{}}, layer_4::@NamedTuple{}, layer_5::@NamedTuple{}}}
└──      return %2


```


Note, that we still recommend using this layer internally and not exposing this as the default API to the users.


<a id='Appendix'></a>

## Appendix


```julia
using InteractiveUtils
InteractiveUtils.versioninfo()
if @isdefined(LuxCUDA) && CUDA.functional(); println(); CUDA.versioninfo(); end
if @isdefined(LuxAMDGPU) && LuxAMDGPU.functional(); println(); AMDGPU.versioninfo(); end
```


```
Julia Version 1.10.2
Commit bd47eca2c8a (2024-03-01 10:14 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 48 × AMD EPYC 7402 24-Core Processor
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-15.0.7 (ORCJIT, znver2)
Threads: 48 default, 0 interactive, 24 GC (on 2 virtual cores)
Environment:
  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64
  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6
  JULIA_PROJECT = /var/lib/buildkite-agent/builds/gpuci-11/julialang/lux-dot-jl/docs/Project.toml
  JULIA_AMDGPU_LOGGING_ENABLED = true
  JULIA_DEBUG = Literate
  JULIA_CPU_THREADS = 2
  JULIA_NUM_THREADS = 48
  JULIA_LOAD_PATH = @:@v#.#:@stdlib
  JULIA_CUDA_HARD_MEMORY_LIMIT = 25%

CUDA runtime 12.3, artifact installation
CUDA driver 12.4
NVIDIA driver 550.54.14

CUDA libraries: 
- CUBLAS: 12.3.4
- CURAND: 10.3.4
- CUFFT: 11.0.12
- CUSOLVER: 11.5.4
- CUSPARSE: 12.2.0
- CUPTI: 21.0.0
- NVML: 12.0.0+550.54.14

Julia packages: 
- CUDA: 5.2.0
- CUDA_Driver_jll: 0.7.0+1
- CUDA_Runtime_jll: 0.11.1+0

Toolchain:
- Julia: 1.10.2
- LLVM: 15.0.7

Environment:
- JULIA_CUDA_HARD_MEMORY_LIMIT: 25%

1 device:
  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 3.443 GiB / 4.750 GiB available)
┌ Warning: LuxAMDGPU is loaded but the AMDGPU is not functional.
└ @ LuxAMDGPU ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/LuxAMDGPU/sGa0S/src/LuxAMDGPU.jl:19

```


---


*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

