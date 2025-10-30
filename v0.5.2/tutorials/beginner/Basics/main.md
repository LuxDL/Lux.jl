


<a id='Julia & Lux for the Uninitiated'></a>

# Julia & Lux for the Uninitiated


This is a quick intro to [Lux](https://github.com/avik-pal/:ux.jl) loosely based on:


1. [PyTorch's tutorial](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html).
2. [Flux's tutorial](https://fluxml.ai/Flux.jl/stable/tutorials/2020-09-15-deep-learning-flux/).
3. [Flax's tutorial](https://flax.readthedocs.io/en/latest/notebooks/jax_for_the_impatient.html).


It introduces basic Julia programming, as well `Zygote`, a source-to-source automatic differentiation (AD) framework in Julia. We'll use these tools to build a very simple neural network. Let's start with importing `Lux.jl`


```julia
using Lux, Random
```


```
  Activating project at `/var/lib/buildkite-agent/builds/gpuci-11/julialang/lux-dot-jl/examples`
```


Now let us control the randomness in our code using proper Pseudo Random Number Generator (PRNG)


```julia
rng = Random.default_rng()
Random.seed!(rng, 0)
```


```
Random.TaskLocalRNG()
```


<a id='Arrays'></a>

## Arrays


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


<a id='CUDA Arrays'></a>

### CUDA Arrays


CUDA functionality is provided separately by the [CUDA.jl package](https://github.com/JuliaGPU/CUDA.jl). If you have a GPU and LuxCUDA is installed, Lux will provide CUDA capabilities. For additional details on backends see the manual section.


You can manually add `CUDA`. Once CUDA is loaded you can move any array to the GPU with the `cu` function (or the `gpu` function exported by `Lux``), and it supports all of the above operations with the same syntax.


```julia
using LuxCUDA
if LuxCUDA.functional()
    x_cu = cu(rand(5, 3))
    @show x_cu
end
```


```
5×3 CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}:
 0.857126  0.681728  0.73806
 0.191956  0.506485  0.622865
 0.857257  0.663036  0.239756
 0.54452   0.503186  0.27993
 0.833518  0.975649  0.967811
```


<a id='(Im)mutability'></a>

## (Im)mutability


Lux as you might have read is [Immutable by convention](http://lux.csail.mit.edu/dev/introduction/overview/#Design-Principles) which means that the core library is built without any form of mutation and all functions are pure. However, we don't enforce it in any form. We do **strongly recommend** that users extending this framework for their respective applications don't mutate their arrays.


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


<a id='Managing Randomness'></a>

## Managing Randomness


We rely on the Julia StdLib `Random` for managing the randomness in our execution. First, we create an PRNG (pseudorandom number generator) and seed it.


```julia
rng = Random.default_rng() # Creates a Xoshiro PRNG
Random.seed!(rng, 0)
```


```
Random.TaskLocalRNG()
```


If we call any function that relies on `rng` and uses it via `randn`, `rand`, etc. `rng` will be mutated. As we have already established we care a lot about immutability, hence we should use `Lux.replicate` on PRNGs before using them.


First, let us run a random number generator 3 times with the `replicate`d rng.


```julia
for i in 1:3
    println("Iteration $i ", rand(Lux.replicate(rng), 10))
end
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


<a id='Automatic Differentiation'></a>

## Automatic Differentiation


Julia has quite a few (maybe too many) AD tools. For the purpose of this tutorial, we will use [AbstractDifferentiation.jl](https://github.com/JuliaDiff/AbstractDifferentiation.jl) which provides a uniform API across multiple AD backends. For the backends we will use:


1. [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) – For Jacobian-Vector Product (JVP)
2. [Zygote.jl](https://github.com/FluxML/Zygote.jl) – For Vector-Jacobian Product (VJP)


*Slight Detour*: We have had several questions regarding if we will be considering any other AD system for the reverse-diff backend. For now we will stick to Zygote.jl, however once we have tested Lux extensively with [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl), we will make the switch.


Even though, theoretically, a VJP (Vector-Jacobian product - reverse autodiff) and a JVP (Jacobian-Vector product - forward-mode autodiff) are similar—they compute a product of a Jacobian and a vector—they differ by the computational complexity of the operation. In short, when you have a large number of parameters (hence a wide matrix), a JVP is less efficient computationally than a VJP, and, conversely, a JVP is more efficient when the Jacobian matrix is a tall matrix.


```julia
using ComponentArrays, ForwardDiff, Zygote
import AbstractDifferentiation as AD
```


```
ERROR: LoadError: ArgumentError: Package FiniteDifferences [26cc04aa-876d-5657-8c51-4c34ba976000] is required but does not seem to be installed:
 - Run `Pkg.instantiate()` to install all recorded dependencies.

Stacktrace:
 [1] _require(pkg::Base.PkgId, env::String)
   @ Base ./loading.jl:1774
 [2] _require_prelocked(uuidkey::Base.PkgId, env::String)
   @ Base ./loading.jl:1660
 [3] macro expansion
   @ ./loading.jl:1648 [inlined]
 [4] macro expansion
   @ ./lock.jl:267 [inlined]
 [5] require(into::Module, mod::Symbol)
   @ Base ./loading.jl:1611
 [6] top-level scope
   @ ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/AbstractDifferentiation/eEkWP/ext/AbstractDifferentiationFiniteDifferencesExt.jl:5
 [7] include
   @ ./Base.jl:457 [inlined]
 [8] include_package_for_output(pkg::Base.PkgId, input::String, depot_path::Vector{String}, dl_load_path::Vector{String}, load_path::Vector{String}, concrete_deps::Vector{Pair{Base.PkgId, UInt128}}, source::String)
   @ Base ./loading.jl:2049
 [9] top-level scope
   @ stdin:3
in expression starting at /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/AbstractDifferentiation/eEkWP/ext/AbstractDifferentiationFiniteDifferencesExt.jl:1
in expression starting at stdin:3
┌ Error: Error during loading of extension AbstractDifferentiationFiniteDifferencesExt of AbstractDifferentiation, use `Base.retry_load_extensions()` to retry.
│   exception =
│    1-element ExceptionStack:
│    Failed to precompile AbstractDifferentiationFiniteDifferencesExt [fa2c61a9-546b-5d5f-9efd-64b8917aca1d] to "/root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/compiled/v1.9/AbstractDifferentiationFiniteDifferencesExt/jl_Bo9Xls".
│    Stacktrace:
│      [1] error(s::String)
│        @ Base ./error.jl:35
│      [2] compilecache(pkg::Base.PkgId, path::String, internal_stderr::IO, internal_stdout::IO, keep_loaded_modules::Bool)
│        @ Base ./loading.jl:2300
│      [3] compilecache
│        @ ./loading.jl:2167 [inlined]
│      [4] _require(pkg::Base.PkgId, env::Nothing)
│        @ Base ./loading.jl:1805
│      [5] _require_prelocked(uuidkey::Base.PkgId, env::Nothing)
│        @ Base ./loading.jl:1660
│      [6] _require_prelocked(uuidkey::Base.PkgId)
│        @ Base ./loading.jl:1658
│      [7] run_extension_callbacks(extid::Base.ExtensionId)
│        @ Base ./loading.jl:1255
│      [8] run_extension_callbacks(pkgid::Base.PkgId)
│        @ Base ./loading.jl:1290
│      [9] run_package_callbacks(modkey::Base.PkgId)
│        @ Base ./loading.jl:1124
│     [10] _require_prelocked(uuidkey::Base.PkgId, env::String)
│        @ Base ./loading.jl:1667
│     [11] macro expansion
│        @ ./loading.jl:1648 [inlined]
│     [12] macro expansion
│        @ ./lock.jl:267 [inlined]
│     [13] require(into::Module, mod::Symbol)
│        @ Base ./loading.jl:1611
│     [14] eval
│        @ ./boot.jl:370 [inlined]
│     [15] #17
│        @ ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/Documenter/bYYzK/src/Expanders.jl:629 [inlined]
│     [16] cd(f::Documenter.Expanders.var"#17#19"{Module, Expr}, dir::String)
│        @ Base.Filesystem ./file.jl:112
│     [17] (::Documenter.Expanders.var"#16#18"{Documenter.Documents.Page, Module, Expr})()
│        @ Documenter.Expanders ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/Documenter/bYYzK/src/Expanders.jl:628
│     [18] (::IOCapture.var"#3#5"{DataType, Documenter.Expanders.var"#16#18"{Documenter.Documents.Page, Module, Expr}, Task, IOContext{Base.PipeEndpoint}, IOContext{Base.PipeEndpoint}, Base.TTY, Base.TTY})()
│        @ IOCapture ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/IOCapture/8Uj7o/src/IOCapture.jl:119
│     [19] with_logstate(f::Function, logstate::Any)
│        @ Base.CoreLogging ./logging.jl:514
│     [20] with_logger
│        @ ./logging.jl:626 [inlined]
│     [21] capture(f::Documenter.Expanders.var"#16#18"{Documenter.Documents.Page, Module, Expr}; rethrow::Type, color::Bool)
│        @ IOCapture ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/IOCapture/8Uj7o/src/IOCapture.jl:116
│     [22] runner(#unused#::Type{Documenter.Expanders.ExampleBlocks}, x::Markdown.Code, page::Documenter.Documents.Page, doc::Documenter.Documents.Document)
│        @ Documenter.Expanders ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/Documenter/bYYzK/src/Expanders.jl:627
│     [23] dispatch(::Type{Documenter.Expanders.ExpanderPipeline}, ::Markdown.Code, ::Vararg{Any})
│        @ Documenter.Utilities.Selectors ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/Documenter/bYYzK/src/Utilities/Selectors.jl:170
│     [24] expand(doc::Documenter.Documents.Document)
│        @ Documenter.Expanders ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/Documenter/bYYzK/src/Expanders.jl:42
│     [25] runner(#unused#::Type{Documenter.Builder.ExpandTemplates}, doc::Documenter.Documents.Document)
│        @ Documenter.Builder ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/Documenter/bYYzK/src/Builder.jl:226
│     [26] dispatch(#unused#::Type{Documenter.Builder.DocumentPipeline}, x::Documenter.Documents.Document)
│        @ Documenter.Utilities.Selectors ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/Documenter/bYYzK/src/Utilities/Selectors.jl:170
│     [27] #2
│        @ ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/Documenter/bYYzK/src/Documenter.jl:273 [inlined]
│     [28] cd(f::Documenter.var"#2#3"{Documenter.Documents.Document}, dir::String)
│        @ Base.Filesystem ./file.jl:112
│     [29] #makedocs#1
│        @ ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/Documenter/bYYzK/src/Documenter.jl:272 [inlined]
│     [30] top-level scope
│        @ /var/lib/buildkite-agent/builds/gpuci-11/julialang/lux-dot-jl/docs/make.jl:12
│     [31] include(fname::String)
│        @ Base.MainInclude ./client.jl:478
│     [32] top-level scope
│        @ none:15
│     [33] eval
│        @ ./boot.jl:370 [inlined]
│     [34] exec_options(opts::Base.JLOptions)
│        @ Base ./client.jl:280
│     [35] _start()
│        @ Base ./client.jl:522
└ @ Base loading.jl:1261
```


<a id='Gradients'></a>

### Gradients


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


Let's use AbstractDifferentiation and Zygote to compute the gradients.


```julia
println("Actual Gradient: ", ∇f(v))
println("Computed Gradient via Reverse Mode AD (Zygote): ",
    AD.gradient(AD.ZygoteBackend(), f, v)[1])
println("Computed Gradient via Forward Mode AD (ForwardDiff): ",
    AD.gradient(AD.ForwardDiffBackend(), f, v)[1])
```


```
Actual Gradient: Float32[-0.4051151, -0.4593922, 0.92155594, 1.1871622]
Computed Gradient via Reverse Mode AD (Zygote): Float32[-0.4051151, -0.4593922, 0.92155594, 1.1871622]
Computed Gradient via Forward Mode AD (ForwardDiff): Float32[-0.4051151, -0.4593922, 0.92155594, 1.1871622]
```


Note that `AD.gradient` will only work for scalar valued outputs.


<a id='Jacobian-Vector Product'></a>

### Jacobian-Vector Product


I will defer the discussion on forward-mode AD to [https://book.sciml.ai/notes/08/](https://book.sciml.ai/notes/08/). Here let us just look at a mini example on how to use it.


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


Construct the pushforward function.


```julia
pf_f = AD.value_and_pushforward_function(AD.ForwardDiffBackend(), f, x)
```


```
#17 (generic function with 1 method)
```


Compute the jvp.


```julia
val, jvp = pf_f(v)
println("Computed Value: f(", x, ") = ", val)
println("JVP: ", jvp[1])
```


```
Computed Value: f(Float32[-0.877497, 1.1953009, -0.057005208, 0.25055695, 0.09351656]) = Float32[0.3850005, 0.71437216, 0.0016247969, 0.031389393, 0.0043726736]
JVP: Float32[-0.877497, 1.1953009, -0.057005208, 0.25055695, 0.09351656]
```


<a id='Vector-Jacobian Product'></a>

### Vector-Jacobian Product


Using the same function and inputs, let us compute the VJP.


```julia
pb_f = AD.value_and_pullback_function(AD.ZygoteBackend(), f, x)
```


```
#25 (generic function with 1 method)
```


Compute the vjp.


```julia
val, vjp = pb_f(v)
println("Computed Value: f(", x, ") = ", val)
println("VJP: ", vjp[1])
```


```
Computed Value: f(Float32[-0.877497, 1.1953009, -0.057005208, 0.25055695, 0.09351656]) = Float32[0.3850005, 0.71437216, 0.0016247969, 0.031389393, 0.0043726736]
VJP: Float32[-0.877497, 1.1953009, -0.057005208, 0.25055695, 0.09351656]
```


<a id='Linear Regression'></a>

## Linear Regression


Finally, now let us consider a linear regression problem. From a set of data-points $\{ (x_i, y_i), i \in \{ 1, \dots, k \}, x_i \in \mathbb{R}^n, y_i \in \mathbb{R}^m \}$, we try to find a set of parameters $W$ and $b$, s.t. $f_{W,b}(x) = Wx + b$, which minimizes the mean squared error:


$$
L(W, b) \longrightarrow \sum_{i = 1}^{k} \frac{1}{2} \| y_i - f_{W,b}(x_i) \|_2^2
$$


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
ps = ps |> ComponentArray
```


```
ComponentVector{Float32}(weight = Float32[-0.5583162 0.3457679 … -0.35419345 0.039559156; -0.05661944 -0.4899126 … 0.22614014 0.27704597; … ; 0.06026341 -0.11202827 … 0.42526972 -0.3576447; 0.23414856 -0.5949539 … 0.08254115 -0.5224755], bias = Float32[0.0; 0.0; … ; 0.0; 0.0;;])
```


Set problem dimensions.


```julia
n_samples = 20
x_dim = 10
y_dim = 5
```


```
5
```


Generate random ground truth W and b.


```julia
W = randn(rng, Float32, y_dim, x_dim)
b = randn(rng, Float32, y_dim)
```


```
5-element Vector{Float32}:
  0.68468636
 -0.57578707
  0.0594993
 -0.9436797
  1.5164032
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
using Optimisers

opt = Optimisers.Descent(0.01f0)
```


```
Optimisers.Descent{Float32}(0.01f0)
```


Initialize the initial state of the optimiser


```julia
opt_state = Optimisers.setup(opt, ps)
```


```
Leaf(Descent{Float32}(0.01), nothing)
```


Define the loss function


```julia
mse(model, ps, st, X, y) = sum(abs2, model(X, ps, st)[1] .- y)
mse(weight, bias, X, y) = sum(abs2, weight * X .+ bias .- y)
loss_function(ps, X, y) = mse(model, ps, st, X, y)

println("Loss Value with ground true parameters: ", mse(W, b, x_samples, y_samples))

for i in 1:100
    # In actual code, don't use globals. But here I will simply for the sake of
    # demonstration
    global ps, st, opt_state
    # Compute the gradient
    gs = gradient(loss_function, ps, x_samples, y_samples)[1]
    # Update model parameters
    opt_state, ps = Optimisers.update(opt_state, ps, gs)
    if i % 10 == 1 || i == 100
        println("Loss Value after $i iterations: ",
            mse(model, ps, st, x_samples, y_samples))
    end
end
```


```
Loss Value with ground true parameters: 0.009175307
Loss Value after 1 iterations: 165.57005
Loss Value after 11 iterations: 4.351237
Loss Value after 21 iterations: 0.6856849
Loss Value after 31 iterations: 0.15421417
Loss Value after 41 iterations: 0.041469414
Loss Value after 51 iterations: 0.014032223
Loss Value after 61 iterations: 0.006883738
Loss Value after 71 iterations: 0.004938521
Loss Value after 81 iterations: 0.004391277
Loss Value after 91 iterations: 0.0042331247
Loss Value after 100 iterations: 0.0041888584
```


---


*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

