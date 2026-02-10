---
url: /dev/tutorials/beginner/2_PolynomialFitting.md
---
# Fitting a Polynomial using MLP {#Fitting-a-Polynomial-using-MLP}

In this tutorial we will fit a MultiLayer Perceptron (MLP) on data generated from a polynomial.

## Package Imports {#Package-Imports}

```julia
using Lux, ADTypes, Optimisers, Printf, Random, Reactant, Statistics, CairoMakie
```

## Dataset {#Dataset}

Generate 128 datapoints from the polynomial $y = x^2 - 2x$.

```julia
function generate_data(rng::AbstractRNG)
    x = reshape(collect(range(-2.0f0, 2.0f0, 128)), (1, 128))
    poly_coeffs = (0, -2, 1)
    y = evalpoly.(x, (poly_coeffs,))
    # add some noise to simulate real-world conditions
    y .+= randn(rng, Float32, (1, 128)) .* 0.1f0
    return (x, y)
end
```

Initialize the random number generator and fetch the dataset.

```julia
rng = MersenneTwister()
Random.seed!(rng, 12345)

(x, y) = generate_data(rng)
```

```
(Float32[-2.0 -1.968504 -1.9370079 -1.9055119 -1.8740157 -1.8425196 -1.8110236 -1.7795275 -1.7480315 -1.7165354 -1.6850394 -1.6535434 -1.6220472 -1.5905511 -1.5590551 -1.527559 -1.496063 -1.464567 -1.4330709 -1.4015749 -1.3700787 -1.3385826 -1.3070866 -1.2755905 -1.2440945 -1.2125984 -1.1811024 -1.1496063 -1.1181102 -1.0866141 -1.0551181 -1.023622 -0.992126 -0.96062994 -0.92913383 -0.8976378 -0.86614174 -0.8346457 -0.8031496 -0.77165353 -0.7401575 -0.70866144 -0.6771653 -0.6456693 -0.61417323 -0.5826772 -0.5511811 -0.51968503 -0.48818898 -0.4566929 -0.42519686 -0.39370078 -0.36220473 -0.33070865 -0.2992126 -0.26771653 -0.23622048 -0.20472442 -0.17322835 -0.14173229 -0.11023622 -0.07874016 -0.047244094 -0.015748031 0.015748031 0.047244094 0.07874016 0.11023622 0.14173229 0.17322835 0.20472442 0.23622048 0.26771653 0.2992126 0.33070865 0.36220473 0.39370078 0.42519686 0.4566929 0.48818898 0.51968503 0.5511811 0.5826772 0.61417323 0.6456693 0.6771653 0.70866144 0.7401575 0.77165353 0.8031496 0.8346457 0.86614174 0.8976378 0.92913383 0.96062994 0.992126 1.023622 1.0551181 1.0866141 1.1181102 1.1496063 1.1811024 1.2125984 1.2440945 1.2755905 1.3070866 1.3385826 1.3700787 1.4015749 1.4330709 1.464567 1.496063 1.527559 1.5590551 1.5905511 1.6220472 1.6535434 1.6850394 1.7165354 1.7480315 1.7795275 1.8110236 1.8425196 1.8740157 1.9055119 1.9370079 1.968504 2.0], Float32[8.080871 7.562357 7.451749 7.5005703 7.295229 7.2245107 6.8731666 6.7092047 6.5385857 6.4631066 6.281978 5.960991 5.963052 5.68927 5.3667717 5.519665 5.2999034 5.0238676 5.174298 4.6706038 4.570324 4.439068 4.4462147 4.299262 3.9799082 3.9492173 3.8747025 3.7264304 3.3844414 3.2934628 3.1180353 3.0698316 3.0491123 2.592982 2.8164148 2.3875027 2.3781595 2.4269633 2.2763796 2.3316176 2.0829067 1.9049499 1.8581494 1.7632381 1.7745113 1.5406592 1.3689325 1.2614254 1.1482575 1.2801026 0.9070533 0.91188717 0.9415703 0.85747254 0.6692604 0.7172643 0.48259094 0.48990166 0.35299227 0.31578436 0.25483933 0.37486005 0.19847682 -0.042415008 -0.05951088 0.014774345 -0.114184186 -0.15978265 -0.29916334 -0.22005874 -0.17161606 -0.3613516 -0.5489093 -0.7267406 -0.5943626 -0.62129945 -0.50063384 -0.6346849 -0.86081326 -0.58715504 -0.5171875 -0.6575044 -0.71243864 -0.78395927 -0.90537953 -0.9515314 -0.8603811 -0.92880917 -1.0078154 -0.90215015 -1.0109437 -1.0764086 -1.1691734 -1.0740278 -1.1429857 -1.104191 -0.948015 -0.9233653 -0.82379496 -0.9810639 -0.92863405 -0.9360056 -0.92652786 -0.847396 -1.115507 -1.0877254 -0.92295444 -0.86975616 -0.81879705 -0.8482455 -0.6524158 -0.6184501 -0.7483137 -0.60395515 -0.67555165 -0.6288941 -0.6774449 -0.49889082 -0.43817532 -0.46497717 -0.30316323 -0.36745527 -0.3227286 -0.20977046 -0.09777648 -0.053120755 -0.15877295 -0.06777584])
```

Let's visualize the dataset

```julia
begin
    fig = Figure()
    ax = CairoMakie.Axis(fig[1, 1]; xlabel="x", ylabel="y")

    l = lines!(
        ax,
        x[1, :],
        x -> evalpoly(x, (0, -2, 1));
        linewidth=3,
        color=:blue,
        label="True Quadratic Function",
    )
    s = scatter!(
        ax,
        x[1, :],
        y[1, :];
        markersize=12,
        alpha=0.5,
        color=:orange,
        strokecolor=:black,
        strokewidth=2,
        label="Actual Data",
    )

    axislegend(ax)

    fig
end
```

## Neural Network {#Neural-Network}

For this problem, you should not be using a neural network. But let's still do that!

```julia
model = Chain(Dense(1 => 16, relu), Dense(16 => 1))
```

```
Chain(
    layer_1 = Dense(1 => 16, relu),               # 32 parameters
    layer_2 = Dense(16 => 1),                     # 17 parameters
)         # Total: 49 parameters,
          #        plus 0 states.
```

## Optimizer {#Optimizer}

We will use Adam from [Optimisers.jl](https://fluxml.ai/Optimisers.jl)

```julia
opt = Adam(0.03f0)
```

```
Optimisers.Adam(eta=0.03, beta=(0.9, 0.999), epsilon=1.0e-8)
```

## Loss Function {#Loss-Function}

We will use the `Training` API so we need to ensure that our loss function takes 4 inputs – model, parameters, states and data. The function must return 3 values – loss, updated\_state, and any computed statistics. This is already satisfied by the loss functions provided by Lux.

```julia
const loss_function = MSELoss()

const cdev = cpu_device()
const xdev = reactant_device()

ps, st = Lux.setup(rng, model) |> xdev
```

```
((layer_1 = (weight = Reactant.ConcretePJRTArray{Float32, 2, 1}(Float32[2.2569513; 1.8385266; 1.8834435; -1.4215803; -0.1289033; -1.4116536; -1.4359436; -2.3610642; -0.847535; 1.6091344; -0.34999675; 1.9372884; -0.41628727; 1.1786895; -1.4312565; 0.34652048;;]), bias = Reactant.ConcretePJRTArray{Float32, 1, 1}(Float32[0.9155488, -0.005158901, 0.5026965, -0.84174657, -0.9167142, -0.14881086, -0.8202727, 0.19286752, 0.60171676, 0.951689, 0.4595859, -0.33281517, -0.692657, 0.4369135, 0.3800323, 0.61768365])), layer_2 = (weight = Reactant.ConcretePJRTArray{Float32, 2, 1}(Float32[0.20061705 0.22529833 0.07667785 0.115506485 0.22827768 0.22680467 0.0035893882 -0.39495495 0.18033011 -0.02850357 -0.08613788 -0.3103005 0.12508307 -0.087390475 -0.13759731 0.08034529]), bias = Reactant.ConcretePJRTArray{Float32, 1, 1}(Float32[0.06066203]))), (layer_1 = NamedTuple(), layer_2 = NamedTuple()))
```

## Training {#Training}

First we will create a [`Training.TrainState`](/api/Lux/utilities#Lux.Training.TrainState) which is essentially a convenience wrapper over parameters, states and optimizer states.

```julia
tstate = Training.TrainState(model, ps, st, opt)
```

```
TrainState(
    Chain(
        layer_1 = Dense(1 => 16, relu),           # 32 parameters
        layer_2 = Dense(16 => 1),                 # 17 parameters
    ),
    number of parameters: 49
    number of states: 0
    optimizer: ReactantOptimiser(Optimisers.Adam(eta=Reactant.ConcretePJRTNumber{Float32, 1}(0.03f0), beta=(Reactant.ConcretePJRTNumber{Float64, 1}(0.9), Reactant.ConcretePJRTNumber{Float64, 1}(0.999)), epsilon=Reactant.ConcretePJRTNumber{Float64, 1}(1.0e-8)))
    step: 0
)

```

Now we will use Enzyme (Reactant) for our AD requirements.

```julia
vjp_rule = AutoEnzyme()
```

Finally the training loop.

```julia
function main(tstate::Training.TrainState, vjp, data, epochs)
    data = xdev(data)
    for epoch in 1:epochs
        _, loss, _, tstate = Training.single_train_step!(vjp, loss_function, data, tstate)
        if epoch % 50 == 1 || epoch == epochs
            @printf "Epoch: %3d \t Loss: %.5g\n" epoch loss
        end
    end
    return tstate
end

tstate = main(tstate, vjp_rule, (x, y), 250)
```

```
TrainState(
    Chain(
        layer_1 = Dense(1 => 16, relu),           # 32 parameters
        layer_2 = Dense(16 => 1),                 # 17 parameters
    ),
    number of parameters: 49
    number of states: 0
    optimizer: ReactantOptimiser(Optimisers.Adam(eta=Reactant.ConcretePJRTNumber{Float32, 1}(0.03f0), beta=(Reactant.ConcretePJRTNumber{Float64, 1}(0.9), Reactant.ConcretePJRTNumber{Float64, 1}(0.999)), epsilon=Reactant.ConcretePJRTNumber{Float64, 1}(1.0e-8)))
    step: 250
    cache: TrainingBackendCache(Lux.Training.ReactantBackend{Static.True, Missing, Nothing, ADTypes.AutoEnzyme{Nothing, Nothing}}(static(true), missing, nothing, ADTypes.AutoEnzyme()))
    objective_function: GenericLossFunction
)

```

Since we are using Reactant, we need to compile the model before we can use it.

```julia
forward_pass = @compile Lux.apply(
    tstate.model, xdev(x), tstate.parameters, Lux.testmode(tstate.states)
)

y_pred =
    forward_pass(tstate.model, xdev(x), tstate.parameters, Lux.testmode(tstate.states)) |>
    first |>
    cdev
```

Let's plot the results

```julia
begin
    fig = Figure()
    ax = CairoMakie.Axis(fig[1, 1]; xlabel="x", ylabel="y")

    l = lines!(
        ax,
        x[1, :],
        x -> evalpoly(x, (0, -2, 1));
        linewidth=3,
        label="True Quadratic Function",
    )
    s1 = scatter!(
        ax,
        x[1, :],
        y[1, :];
        markersize=12,
        alpha=0.5,
        color=:orange,
        strokecolor=:black,
        strokewidth=2,
        label="Actual Data",
    )
    s2 = scatter!(
        ax,
        x[1, :],
        y_pred[1, :];
        markersize=12,
        alpha=0.5,
        color=:green,
        strokecolor=:black,
        strokewidth=2,
        label="Predictions",
    )

    axislegend(ax)

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
