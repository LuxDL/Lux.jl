
<a id='Quickstart'></a>

# Quickstart


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
model = Chain(BatchNorm(128), Dense(128, 256, tanh), BatchNorm(256),
    Chain(Dense(256, 1, tanh), Dense(1, 10)))
```


```
Chain(
    layer_1 = BatchNorm(128, affine=true, track_stats=true),  # 256 parameters, plus 257
    layer_2 = Dense(128 => 256, tanh_fast),  # 33_024 parameters
    layer_3 = BatchNorm(256, affine=true, track_stats=true),  # 512 parameters, plus 513
    layer_4 = Dense(256 => 1, tanh_fast),  # 257 parameters
    layer_5 = Dense(1 => 10),           # 20 parameters
)         # Total: 34_069 parameters,
          #        plus 770 states.
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
((layer_1 = (scale = Leaf(Adam{Float32}(0.0001, (0.9, 0.999), 1.19209f-7), (Float32[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], Float32[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (0.81, 0.998001))), bias = Leaf(Adam{Float32}(0.0001, (0.9, 0.999), 1.19209f-7), (Float32[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], Float32[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (0.81, 0.998001)))), layer_2 = (weight = Leaf(Adam{Float32}(0.0001, (0.9, 0.999), 1.19209f-7), (Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], (0.81, 0.998001))), bias = Leaf(Adam{Float32}(0.0001, (0.9, 0.999), 1.19209f-7), (Float32[0.0; 0.0; … ; 0.0; 0.0;;], Float32[0.0; 0.0; … ; 0.0; 0.0;;], (0.81, 0.998001)))), layer_3 = (scale = Leaf(Adam{Float32}(0.0001, (0.9, 0.999), 1.19209f-7), (Float32[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], Float32[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (0.81, 0.998001))), bias = Leaf(Adam{Float32}(0.0001, (0.9, 0.999), 1.19209f-7), (Float32[0.000865326, -0.00166989, 0.00235583, 5.5809f-6, -0.00181531, -0.00236466, -0.000731729, 0.00176976, 0.00116876, 0.00076472  …  -0.00168781, 0.000968752, -0.000790557, 0.0018812, 7.08135f-5, -0.00192429, 0.00018412, 0.000386612, -0.000819421, 0.000331427], Float32[7.48779f-8, 2.7885f-7, 5.54987f-7, 3.1146f-12, 3.2953f-7, 5.59154f-7, 5.35419f-8, 3.13202f-7, 1.36597f-7, 5.84788f-8  …  2.84867f-7, 9.38468f-8, 6.24972f-8, 3.53887f-7, 5.01448f-10, 3.70285f-7, 3.38997f-9, 1.49467f-8, 6.71442f-8, 1.09842f-8], (0.81, 0.998001)))), layer_4 = (weight = Leaf(Adam{Float32}(0.0001, (0.9, 0.999), 1.19209f-7), (Float32[0.0 0.0 … 0.0 0.0], Float32[0.0 0.0 … 0.0 0.0], (0.81, 0.998001))), bias = Leaf(Adam{Float32}(0.0001, (0.9, 0.999), 1.19209f-7), (Float32[0.0160788;;], Float32[2.58524f-5;;], (0.81, 0.998001)))), layer_5 = (weight = Leaf(Adam{Float32}(0.0001, (0.9, 0.999), 1.19209f-7), (Float32[0.0; 0.0; … ; 0.0; 0.0;;], Float32[0.0; 0.0; … ; 0.0; 0.0;;], (0.81, 0.998001))), bias = Leaf(Adam{Float32}(0.0001, (0.9, 0.999), 1.19209f-7), (Float32[0.2; 0.2; … ; 0.2; 0.2;;], Float32[0.00399995; 0.00399995; … ; 0.00399995; 0.00399995;;], (0.81, 0.998001))))), (layer_1 = (scale = Float32[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0  …  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], bias = Float32[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), layer_2 = (weight = Float32[-0.11034693 0.10973185 … 0.097955346 -0.009067461; -0.0111903995 0.07578978 … -0.03190492 0.08886787; … ; 0.01854451 -0.035003364 … -0.016294405 0.019076452; -0.09206565 -0.047390625 … -0.08859007 0.009517342], bias = Float32[0.0; 0.0; … ; 0.0; 0.0;;]), layer_3 = (scale = Float32[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0  …  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], bias = Float32[-9.999862f-5, 9.9999284f-5, -9.999949f-5, -9.978685f-5, 9.999934f-5, 9.999949f-5, 9.999837f-5, -9.999932f-5, -9.9998986f-5, -9.999843f-5  …  9.999929f-5, -9.999877f-5, 9.999849f-5, -9.9999364f-5, -9.9983175f-5, 9.999938f-5, -9.999352f-5, -9.999691f-5, 9.999854f-5, -9.999641f-5]), layer_4 = (weight = Float32[0.05381791 -0.103856824 … -0.050962884 0.020612676], bias = Float32[-9.9999925f-5;;]), layer_5 = (weight = Float32[-0.65478534; 0.61009777; … ; 0.41110995; 0.5493141;;], bias = Float32[-0.0001; -0.0001; … ; -0.0001; -0.0001;;])))
```

