
<a id='Getting-Started'></a>

# Getting Started


<a id='Installation'></a>

## Installation


Install [Julia v1.6 or above](https://julialang.org/downloads/). Lux.jl is available through the Julia package manager. You can enter it by pressing `]` in the REPL and then typing


```julia
pkg> add Lux
```


Alternatively, you can also do


```julia
import Pkg; Pkg.add("Lux")
```


:::tip


The Julia Compiler is always improving. As such, we recommend using the latest stable version of Julia instead of the LTS.


:::


<a id='Quickstart'></a>

## Quickstart


:::tip PRE-REQUISITES


You need to install `Optimisers` and `Zygote` if not done already. `Pkg.add(["Optimisers", "Zygote"])`


:::


```julia
using Lux, Random, Optimisers, Zygote
# using LuxCUDA, LuxAMDGPU, Metal # Optional packages for GPU support
```


We take randomness very seriously


```julia
# Seeding
rng = Random.default_rng()
Random.seed!(rng, 0)
```


```
Random.TaskLocalRNG()
```


Build the model


```julia
# Construct the layer
model = Chain(Dense(128, 256, tanh), Chain(Dense(256, 1, tanh), Dense(1, 10)))
```


```
Chain(
    layer_1 = Dense(128 => 256, tanh_fast),  # 33_024 parameters
    layer_2 = Dense(256 => 1, tanh_fast),  # 257 parameters
    layer_3 = Dense(1 => 10),           # 20 parameters
)         # Total: 33_301 parameters,
          #        plus 0 states.
```


Models don't hold parameters and states so initialize them. From there on, we just use our standard AD and Optimisers API.


```julia
# Get the device determined by Lux
device = gpu_device()

# Parameter and State Variables
ps, st = Lux.setup(rng, model) .|> device

# Dummy Input
x = rand(rng, Float32, 128, 2) |> device

# Run the model
y, st = Lux.apply(model, x, ps, st)

# Gradients
## Pullback API to capture change in state
(l, st_), pb = pullback(p -> Lux.apply(model, x, p, st), ps)
gs = pb((one.(l), nothing))[1]

# Optimization
st_opt = Optimisers.setup(Adam(0.0001f0), ps)
st_opt, ps = Optimisers.update(st_opt, ps, gs)
```


```
((layer_1 = (weight = Leaf(Adam(0.0001, (0.9, 0.999), 1.0e-8), (Float32[0.00313608 0.00806096 … 0.00476192 0.00732118; -0.00447309 -0.0119719 … -0.00822211 -0.0110335; … ; -0.00294453 -0.00749935 … -0.00426221 -0.00678769; 0.000750543 0.00195163 … 0.00120731 0.00178011], Float32[9.83485f-7 6.49782f-6 … 2.26756f-6 5.3599f-6; 2.00083f-6 1.43324f-5 … 6.76022f-6 1.21738f-5; … ; 8.67016f-7 5.62395f-6 … 1.81662f-6 4.60721f-6; 5.63307f-8 3.80882f-7 … 1.45758f-7 3.16876f-7], (0.81, 0.998001))), bias = Leaf(Adam(0.0001, (0.9, 0.999), 1.0e-8), (Float32[0.00954525; -0.0146331; … ; -0.00881351; 0.00233261;;], Float32[9.11106f-6; 2.14125f-5; … ; 7.76769f-6; 5.44098f-7;;], (0.81, 0.998001)))), layer_2 = (weight = Leaf(Adam(0.0001, (0.9, 0.999), 1.0e-8), (Float32[-0.0104967 0.0714637 … -0.0224641 0.108277], Float32[1.10179f-5 0.000510699 … 5.04627f-5 0.00117238], (0.81, 0.998001))), bias = Leaf(Adam(0.0001, (0.9, 0.999), 1.0e-8), (Float32[0.178909;;], Float32[0.0032008;;], (0.81, 0.998001)))), layer_3 = (weight = Leaf(Adam(0.0001, (0.9, 0.999), 1.0e-8), (Float32[-0.105128; -0.105128; … ; -0.105128; -0.105128;;], Float32[0.00110518; 0.00110518; … ; 0.00110518; 0.00110518;;], (0.81, 0.998001))), bias = Leaf(Adam(0.0001, (0.9, 0.999), 1.0e-8), (Float32[0.2; 0.2; … ; 0.2; 0.2;;], Float32[0.00399995; 0.00399995; … ; 0.00399995; 0.00399995;;], (0.81, 0.998001))))), (layer_1 = (weight = Float32[-0.11044693 0.10963185 … 0.097855344 -0.009167461; -0.0110904 0.07588978 … -0.03180492 0.088967875; … ; 0.01864451 -0.034903362 … -0.016194405 0.019176451; -0.09216565 -0.047490627 … -0.08869007 0.009417342], bias = Float32[-9.999999f-5; 9.999998f-5; … ; 9.999999f-5; -9.9999954f-5;;]), layer_2 = (weight = Float32[0.05391791 -0.103956826 … -0.050862882 0.020512676], bias = Float32[-0.0001;;]), layer_3 = (weight = Float32[-0.6546853; 0.6101978; … ; 0.41120994; 0.5494141;;], bias = Float32[-0.0001; -0.0001; … ; -0.0001; -0.0001;;])))
```


<a id='Defining-Custom-Layers'></a>

## Defining Custom Layers


```julia
using Lux, Random, Optimisers, Zygote
# using LuxCUDA, LuxAMDGPU, Metal # Optional packages for GPU support
import Lux.Experimental: @compact
```


We will define a custom MLP using the `@compact` macro. The macro takes in a list of parameters, layers and states, and a function defining the forward pass of the neural network.


```julia
n_in = 1
n_out = 1
nlayers = 3

model = @compact(w1=Dense(n_in, 128),
    w2=[Dense(128, 128) for i in 1:nlayers],
    w3=Dense(128, n_out),
    act=relu) do x
    embed = act(w1(x))
    for w in w2
        embed = act(w(embed))
    end
    out = w3(embed)
    return out
end
```


```
@compact(
    w1 = Dense(1 => 128),               # 256 parameters
    w2 = NamedTuple(
        1 = Dense(128 => 128),          # 16_512 parameters
        2 = Dense(128 => 128),          # 16_512 parameters
        3 = Dense(128 => 128),          # 16_512 parameters
    ),
    w3 = Dense(128 => 1),               # 129 parameters
    act = relu,
) do x 
    embed = act(w1(x))
    for w = w2
        embed = act(w(embed))
    end
    out = w3(embed)
    return out
end       # Total: 49_921 parameters,
          #        plus 1 states.
```


We can initialize the model and train it with the same code as before!


```julia
ps, st = Lux.setup(Xoshiro(0), model)

model(randn(n_in, 32), ps, st)  # 1×32 Matrix as output.

x_data = collect(-2.0f0:0.1f0:2.0f0)'
y_data = 2 .* x_data .- x_data .^ 3
st_opt = Optimisers.setup(Adam(), ps)

for epoch in 1:1000
    global st  # Put this in a function in real use-cases
    (loss, st), pb = Zygote.pullback(ps) do p
        y, st_ = model(x_data, p, st)
        return sum(abs2, y .- y_data), st_
    end
    gs = only(pb((one(loss), nothing)))
    epoch % 100 == 1 && println("Epoch: $(epoch) | Loss: $(loss)")
    Optimisers.update!(st_opt, ps, gs)
end
```


```
Epoch: 1 | Loss: 84.32512
Epoch: 101 | Loss: 0.08861052
Epoch: 201 | Loss: 0.007037298
Epoch: 301 | Loss: 0.005391656
Epoch: 401 | Loss: 0.014058021
Epoch: 501 | Loss: 0.0022117028
Epoch: 601 | Loss: 0.0015865607
Epoch: 701 | Loss: 0.21984956
Epoch: 801 | Loss: 0.00019668281
Epoch: 901 | Loss: 0.0018975141
```


<a id='Additional-Packages'></a>

## Additional Packages


`LuxDL` hosts various packages that provide additional functionality for Lux.jl. All packages mentioned in this documentation are available via the Julia General Registry.


You can install all those packages via `import Pkg; Pkg.add(<package name>)`.


<a id='GPU-Support'></a>

## GPU Support


GPU Support for Lux.jl requires loading additional packages:


  * [`LuxCUDA.jl`](https://github.com/LuxDL/LuxCUDA.jl) for CUDA support.
  * [`LuxAMDGPU.jl`](https://github.com/LuxDL/LuxAMDGPU.jl) for AMDGPU support.
  * [`Metal.jl`](https://github.com/JuliaGPU/Metal.jl) for Apple Metal support.

