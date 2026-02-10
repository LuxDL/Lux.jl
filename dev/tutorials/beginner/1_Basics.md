---
url: /dev/tutorials/beginner/1_Basics.md
---
# Julia & Lux for the Uninitiated {#Julia-and-Lux-for-the-Uninitiated}

This is a quick intro to [Lux](https://github.com/LuxDL/Lux.jl) loosely based on:

1. [PyTorch's tutorial](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html).

2. Flux's tutorial (the link for which has now been lost to abyss).

3. [Jax's tutorial](https://jax.readthedocs.io/en/latest/jax-101/index.html).

It introduces basic Julia programming, as well `Zygote`, a source-to-source automatic differentiation (AD) framework in Julia. We'll use these tools to build a very simple neural network. Let's start with importing `Lux.jl`

```julia
using Lux, Random
```

Now let us control the randomness in our code using proper Pseudo Random Number Generator (PRNG)

```julia
rng = Random.default_rng()
Random.seed!(rng, 0)
```

```
Random.TaskLocalRNG()
```

## Arrays {#Arrays}

The starting point for all of our models is the `Array` (sometimes referred to as a `Tensor` in other frameworks). This is really just a list of numbers, which might be arranged into a shape like a square. Let's write down an array with three elements.

```julia
x = [1, 2, 3]
```

```
3-element Vector{Int64}:
 1
 2
 3
```

Here's a matrix – a square array with four elements.

```julia
x = [1 2; 3 4]
```

```
2×2 Matrix{Int64}:
 1  2
 3  4
```

We often work with arrays of thousands of elements, and don't usually write them down by hand. Here's how we can create an array of 5×3 = 15 elements, each a random number from zero to one.

```julia
x = rand(rng, 5, 3)
```

```
5×3 Matrix{Float64}:
 0.455238   0.746943   0.193291
 0.547642   0.746801   0.116989
 0.773354   0.97667    0.899766
 0.940585   0.0869468  0.422918
 0.0296477  0.351491   0.707534
```

There's a few functions like this; try replacing `rand` with `ones`, `zeros`, or `randn`.

By default, Julia works stores numbers is a high-precision format called `Float64`. In ML we often don't need all those digits, and can ask Julia to work with `Float32` instead. We can even ask for more digits using `BigFloat`.

```julia
x = rand(BigFloat, 5, 3)
```

```
5×3 Matrix{BigFloat}:
 0.981339    0.793159  0.459019
 0.043883    0.624384  0.56055
 0.164786    0.524008  0.0355555
 0.414769    0.577181  0.621958
 0.00823197  0.30215   0.655881
```

```julia
x = rand(Float32, 5, 3)
```

```
5×3 Matrix{Float32}:
 0.567794   0.369178   0.342539
 0.0985227  0.201145   0.587206
 0.776598   0.148248   0.0851708
 0.723731   0.0770206  0.839303
 0.404728   0.230954   0.679087
```

We can ask the array how many elements it has.

```julia
length(x)
```

```
15
```

Or, more specifically, what size it has.

```julia
size(x)
```

```
(5, 3)
```

We sometimes want to see some elements of the array on their own.

```julia
x
```

```
5×3 Matrix{Float32}:
 0.567794   0.369178   0.342539
 0.0985227  0.201145   0.587206
 0.776598   0.148248   0.0851708
 0.723731   0.0770206  0.839303
 0.404728   0.230954   0.679087
```

```julia
x[2, 3]
```

```
0.58720636f0
```

This means get the second row and the third column. We can also get every row of the third column.

```julia
x[:, 3]
```

```
5-element Vector{Float32}:
 0.34253937
 0.58720636
 0.085170805
 0.8393034
 0.67908657
```

We can add arrays, and subtract them, which adds or subtracts each element of the array.

```julia
x + x
```

```
5×3 Matrix{Float32}:
 1.13559   0.738356  0.685079
 0.197045  0.40229   1.17441
 1.5532    0.296496  0.170342
 1.44746   0.154041  1.67861
 0.809456  0.461908  1.35817
```

```julia
x - x
```

```
5×3 Matrix{Float32}:
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
```

Julia supports a feature called *broadcasting*, using the `.` syntax. This tiles small arrays (or single numbers) to fill bigger ones.

```julia
x .+ 1
```

```
5×3 Matrix{Float32}:
 1.56779  1.36918  1.34254
 1.09852  1.20114  1.58721
 1.7766   1.14825  1.08517
 1.72373  1.07702  1.8393
 1.40473  1.23095  1.67909
```

We can see Julia tile the column vector `1:5` across all rows of the larger array.

```julia
zeros(5, 5) .+ (1:5)
```

```
5×5 Matrix{Float64}:
 1.0  1.0  1.0  1.0  1.0
 2.0  2.0  2.0  2.0  2.0
 3.0  3.0  3.0  3.0  3.0
 4.0  4.0  4.0  4.0  4.0
 5.0  5.0  5.0  5.0  5.0
```

The x' syntax is used to transpose a column `1:5` into an equivalent row, and Julia will tile that across columns.

```julia
zeros(5, 5) .+ (1:5)'
```

```
5×5 Matrix{Float64}:
 1.0  2.0  3.0  4.0  5.0
 1.0  2.0  3.0  4.0  5.0
 1.0  2.0  3.0  4.0  5.0
 1.0  2.0  3.0  4.0  5.0
 1.0  2.0  3.0  4.0  5.0
```

We can use this to make a times table.

```julia
(1:5) .* (1:5)'
```

```
5×5 Matrix{Int64}:
 1   2   3   4   5
 2   4   6   8  10
 3   6   9  12  15
 4   8  12  16  20
 5  10  15  20  25
```

Finally, and importantly for machine learning, we can conveniently do things like matrix multiply.

```julia
W = randn(5, 10)
x = rand(10)
W * x
```

```
5-element Vector{Float64}:
  1.2197981041108443
 -2.62625877100596
 -2.8573820474674845
 -2.4319346874291314
  1.0108668577150213
```

Julia's arrays are very powerful, and you can learn more about what they can do [here](https://docs.julialang.org/en/v1/manual/arrays/).

### CUDA Arrays {#CUDA-Arrays}

CUDA functionality is provided separately by the [CUDA.jl package](https://github.com/JuliaGPU/CUDA.jl). If you have a GPU and LuxCUDA is installed, Lux will provide CUDA capabilities. For additional details on backends see the manual section.

You can manually add `CUDA`. Once CUDA is loaded you can move any array to the GPU with the `cu` function (or the `gpu` function exported by `Lux`), and it supports all of the above operations with the same syntax.

```julia
using LuxCUDA

if LuxCUDA.functional()
    x_cu = cu(rand(5, 3))
    @show x_cu
end
```

## (Im)mutability {#Immutability}

Lux as you might have read is "Immutable by convention," which means that the core library is built without any form of mutation and all functions are pure. However, we don't enforce it in any form. We do **strongly recommend** that users extending this framework for their respective applications don't mutate their arrays.

```julia
x = reshape(1:8, 2, 4)
```

```
2×4 reshape(::UnitRange{Int64}, 2, 4) with eltype Int64:
 1  3  5  7
 2  4  6  8
```

To update this array, we should first copy the array.

```julia
x_copy = copy(x)
view(x_copy, :, 1) .= 0

println("Original Array ", x)
println("Mutated Array ", x_copy)
```

```
Original Array [1 3 5 7; 2 4 6 8]
Mutated Array [0 3 5 7; 0 4 6 8]

```

Note that our current default AD engine (Zygote) is unable to differentiate through this mutation, however, for these specialized cases it is quite trivial to write custom backward passes. (This problem will be fixed once we move towards Enzyme.jl)

## Managing Randomness {#Managing-Randomness}

We rely on the Julia StdLib `Random` for managing the randomness in our execution. First, we create an PRNG (pseudorandom number generator) and seed it.

```julia
rng = Xoshiro(0)     # Creates a Xoshiro PRNG with seed 0
```

```
Random.Xoshiro(0xdb2fa90498613fdf, 0x48d73dc42d195740, 0x8c49bc52dc8a77ea, 0x1911b814c02405e8, 0x22a21880af5dc689)
```

If we call any function that relies on `rng` and uses it via `randn`, `rand`, etc. `rng` will be mutated. As we have already established we care a lot about immutability, hence we should use `Lux.replicate` on PRNGs before using them.

First, let us run a random number generator 3 times with the `replicate`d rng.

```julia
random_vectors = Vector{Vector{Float64}}(undef, 3)
for i in 1:3
    random_vectors[i] = rand(Lux.replicate(rng), 10)
    println("Iteration $i ", random_vectors[i])
end
@assert random_vectors[1] ≈ random_vectors[2] ≈ random_vectors[3]
```

```
Iteration 1 [0.4552384158732863, 0.5476424498276177, 0.7733535276924052, 0.9405848223512736, 0.02964765308691042, 0.74694291453392, 0.7468008914093891, 0.9766699015845924, 0.08694684883050086, 0.35149138733595564]
Iteration 2 [0.4552384158732863, 0.5476424498276177, 0.7733535276924052, 0.9405848223512736, 0.02964765308691042, 0.74694291453392, 0.7468008914093891, 0.9766699015845924, 0.08694684883050086, 0.35149138733595564]
Iteration 3 [0.4552384158732863, 0.5476424498276177, 0.7733535276924052, 0.9405848223512736, 0.02964765308691042, 0.74694291453392, 0.7468008914093891, 0.9766699015845924, 0.08694684883050086, 0.35149138733595564]

```

As expected we get the same output. We can remove the `replicate` call and we will get different outputs.

```julia
for i in 1:3
    println("Iteration $i ", rand(rng, 10))
end
```

```
Iteration 1 [0.4552384158732863, 0.5476424498276177, 0.7733535276924052, 0.9405848223512736, 0.02964765308691042, 0.74694291453392, 0.7468008914093891, 0.9766699015845924, 0.08694684883050086, 0.35149138733595564]
Iteration 2 [0.018743665453639813, 0.8601828553599953, 0.6556360448565952, 0.7746656838366666, 0.7817315740767116, 0.5553797706980106, 0.1261990389976131, 0.4488101521328277, 0.624383955429775, 0.05657739601024536]
Iteration 3 [0.19597391412112541, 0.6830945313415872, 0.6776220912718907, 0.6456416023530093, 0.6340362477836592, 0.5595843665394066, 0.5675557670686644, 0.34351700231383653, 0.7237308297251812, 0.3691778381831775]

```

## Automatic Differentiation {#Automatic-Differentiation}

Julia has quite a few (maybe too many) AD tools. For the purpose of this tutorial, we will use:

1. [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) – For Jacobian-Vector Product (JVP)

2. [Zygote.jl](https://github.com/FluxML/Zygote.jl) – For Vector-Jacobian Product (VJP)

*Slight Detour*: We have had several questions regarding if we will be considering any other AD system for the reverse-diff backend. For now we will stick to Zygote.jl, however once we have tested Lux extensively with [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl), we will make the switch.

Even though, theoretically, a VJP (Vector-Jacobian product - reverse autodiff) and a JVP (Jacobian-Vector product - forward-mode autodiff) are similar—they compute a product of a Jacobian and a vector—they differ by the computational complexity of the operation. In short, when you have a large number of parameters (hence a wide matrix), a JVP is less efficient computationally than a VJP, and, conversely, a JVP is more efficient when the Jacobian matrix is a tall matrix.

```julia
using ComponentArrays, ForwardDiff, Zygote
```

```
Precompiling packages...
    381.9 ms  ✓ StructUtilsTablesExt (serial)
  1 dependency successfully precompiled in 0 seconds

```

### Gradients {#Gradients}

For our first example, consider a simple function computing $f(x) = \frac{1}{2}x^T x$, where $\nabla f(x) = x$

```julia
f(x) = x' * x / 2
∇f(x) = x  # `∇` can be typed as `\nabla<TAB>`
v = randn(rng, Float32, 4)
```

```
4-element Vector{Float32}:
 -0.4051151
 -0.4593922
  0.92155594
  1.1871622
```

Let's use ForwardDiff and Zygote to compute the gradients.

```julia
println("Actual Gradient: ", ∇f(v))
println("Computed Gradient via Reverse Mode AD (Zygote): ", only(Zygote.gradient(f, v)))
println("Computed Gradient via Forward Mode AD (ForwardDiff): ", ForwardDiff.gradient(f, v))
```

```
Actual Gradient: Float32[-0.4051151, -0.4593922, 0.92155594, 1.1871622]
Computed Gradient via Reverse Mode AD (Zygote): Float32[-0.4051151, -0.4593922, 0.92155594, 1.1871622]
Computed Gradient via Forward Mode AD (ForwardDiff): Float32[-0.4051151, -0.4593922, 0.92155594, 1.1871622]

```

Note that `AD.gradient` will only work for scalar valued outputs.

### Jacobian-Vector Product {#Jacobian-Vector-Product}

I will defer the discussion on forward-mode AD to <https://book.sciml.ai/notes/08-Forward-Mode_Automatic_Differentiation_(AD)_via_High_Dimensional_Algebras/>. Here let us just look at a mini example on how to use it.

```julia
f(x) = x .* x ./ 2
x = randn(rng, Float32, 5)
v = ones(Float32, 5)
```

```
5-element Vector{Float32}:
 1.0
 1.0
 1.0
 1.0
 1.0
```

::: warning Using DifferentiationInterface

While DifferentiationInterface provides these functions for a wider range of backends, we currently don't recommend using them with Lux models, since the functions presented here come with additional goodies like [fast second-order derivatives](/manual/nested_autodiff#nested_autodiff).

:::

Compute the JVP. `AutoForwardDiff` specifies that we want to use `ForwardDiff.jl` for the Jacobian-Vector Product

```julia
jvp = jacobian_vector_product(f, AutoForwardDiff(), x, v)
println("JVP: ", jvp)
```

```
JVP: Float32[-0.877497, 1.1953009, -0.057005208, 0.25055695, 0.09351656]

```

### Vector-Jacobian Product {#Vector-Jacobian-Product}

Using the same function and inputs, let us compute the Vector-Jacobian Product (VJP).

```julia
vjp = vector_jacobian_product(f, AutoZygote(), x, v)
println("VJP: ", vjp)
```

```
VJP: Float32[-0.877497, 1.1953009, -0.057005208, 0.25055695, 0.09351656]

```

## Linear Regression {#Linear-Regression}

Finally, now let us consider a linear regression problem. From a set of data-points ${ (x\_i, y\_i), i \in { 1, \dots, k }, x\_i \in \mathbb{R}^n, y\_i \in \mathbb{R}^m }$, we try to find a set of parameters $W$ and $b$, such that $f\_{W,b}(x) = Wx + b$, which minimizes the mean squared error:

$$L(W, b) \longrightarrow \sum\_{i = 1}^{k} \frac{1}{2} | y\_i - f\_{W,b}(x\_i) |\_2^2$$

We can write `f` from scratch, but to demonstrate `Lux`, let us use the `Dense` layer.

```julia
model = Dense(10 => 5)

rng = Random.default_rng()
Random.seed!(rng, 0)
```

```
Random.TaskLocalRNG()
```

Let us initialize the parameters and states (in this case it is empty) for the model.

```julia
ps, st = Lux.setup(rng, model)
ps = ComponentArray(ps)
```

```
ComponentVector{Float32}(weight = Float32[-0.48351604 0.29944378 0.44048923 0.52216566 0.20001544 0.14378412 4.831728f-6 0.53108513 -0.30674055 0.034259237; -0.049033877 -0.42427677 0.27051237 0.40789896 -0.43846488 -0.17706363 -0.032581452 0.46514037 0.19584312 0.23992884; 0.4501613 0.48263645 -0.29908532 -0.18695378 -0.110237636 -0.44184566 0.40354213 0.25278288 0.18056089 -0.35231933; 0.05218965 -0.09701933 0.27035677 0.12589002 -0.2956183 0.34717596 -0.421895 -0.1307367 0.36829442 -0.30972943; 0.20277861 -0.5152452 -0.22635894 0.18841727 0.29828638 0.21690919 -0.04265763 -0.41919124 0.07148273 -0.45247707], bias = Float32[-0.04199602, -0.093925126, -0.0007736237, -0.19397983, 0.0066712513])
```

Set problem dimensions.

```julia
n_samples = 20
x_dim = 10
y_dim = 5
```

We're going to generate a random set of weights `W` and biases `b` that will act as our true model (also known as the ground truth). The neural network we'll train will be to try and approximate `W` and `b` from example data.

```julia
W = randn(rng, Float32, y_dim, x_dim)
b = randn(rng, Float32, y_dim)
```

Generate samples with additional noise.

```julia
x_samples = randn(rng, Float32, x_dim, n_samples)
y_samples = W * x_samples .+ b .+ 0.01f0 .* randn(rng, Float32, y_dim, n_samples)
println("x shape: ", size(x_samples), "; y shape: ", size(y_samples))
```

```
x shape: (10, 20); y shape: (5, 20)

```

For updating our parameters let's use [Optimisers.jl](https://github.com/FluxML/Optimisers.jl). We will use Stochastic Gradient Descent (SGD) with a learning rate of `0.01`.

```julia
using Optimisers, Printf
```

Define the loss function

```julia
lossfn = MSELoss()

println("Loss Value with ground true parameters: ", lossfn(W * x_samples .+ b, y_samples))
```

```
Loss Value with ground true parameters: 9.3742405e-5

```

We will train the model using our training API.

```julia
function train_model!(model, ps, st, opt, nepochs::Int)
    tstate = Training.TrainState(model, ps, st, opt)
    for i in 1:nepochs
        grads, loss, _, tstate = Training.single_train_step!(
            AutoZygote(), lossfn, (x_samples, y_samples), tstate
        )
        if i == 1 || i % 1000 == 0 || i == nepochs
            @printf "Loss Value after %6d iterations: %.8f\n" i loss
        end
    end
    return tstate.model, tstate.parameters, tstate.states
end

model, ps, st = train_model!(model, ps, st, Descent(0.01f0), 10000)

println("Loss Value after training: ", lossfn(first(model(x_samples, ps, st)), y_samples))
```

```
Loss Value after      1 iterations: 7.80465460
Loss Value after   1000 iterations: 0.12503763
Loss Value after   2000 iterations: 0.02538623
Loss Value after   3000 iterations: 0.00914946
Loss Value after   4000 iterations: 0.00407888
Loss Value after   5000 iterations: 0.00198553
Loss Value after   6000 iterations: 0.00101213
Loss Value after   7000 iterations: 0.00053365
Loss Value after   8000 iterations: 0.00029220
Loss Value after   9000 iterations: 0.00016886
Loss Value after  10000 iterations: 0.00010551
Loss Value after training: 0.00010546902

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
