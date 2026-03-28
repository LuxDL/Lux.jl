---
url: /dev/api/NN_Primitives/LuxLib.md
---
# LuxLib {#LuxLib-API}

Backend for Lux.jl

## Apply Activation {#Apply-Activation}

```julia
fast_activation(σ::F, x::AbstractArray) where {F}
```

Compute `σ.(x)` with the best possible implementation available. On CPUs we unroll the loop and use LoopVectorization.jl to vectorize the computation. On GPUs we use simply use broadcasting.

::: tip Note

This function doesn't replace `σ` with `NNlib.fast_act(σ, ...)`, that needs to be done by the user if needed.

:::

**Arguments**

* `σ`: Activation function

* `x`: Input array

**Returns**

* Output Array with the same size as `x`

source

```julia
fast_activation!!(σ::F, x::AbstractArray) where {F}
```

Compute `σ.(x)` with the best possible implementation available. If it is possible to rewrite `x` in-place, it does so. If `x` is an immutable array, it falls back to the generic implementation.

::: tip Note

This function doesn't replace `σ` with `NNlib.fast_act(σ, ...)`, that needs to be done by the user if needed.

:::

::: tip Load `SLEEFPirates.jl` to get faster activations

Certain activation functions are replaced with specialized implementations from [SLEEFPirates.jl](https://github.com/JuliaSIMD/SLEEFPirates.jl) for FP32. This might lead to faster performance but can cause slight decrease in accuracy (in the floating point limit).

:::

**Arguments**

* `σ`: Activation function

* `x`: Input array

**Returns**

* Output Array with the same size as `x`

source

## Attention {#Attention}

```julia
scaled_dot_product_attention(
    query, key, value; head_dim=1, token_dim=3,
    scale=nothing, mask=nothing, is_causal=nothing, bias=nothing,
    fdrop=identity
)
```

Compute scaled dot-product attention for general tensors.

This function is a thin API wrapper that forwards to the implementation in `impl/attention.jl` which supports flexible placement of the `head` and `token` dimensions (via `head_dim` and `token_dim`) and additional features such as grouped key/value (KV) sharing, optional masking, and dropout on the attention weights.

**Arguments**

* `query`, `key`, `value`: N‑dimensional arrays (N ≥ 4). The first three logical dimensions correspond to an embedding/head dimension, a heads dimension, and a token (sequence) dimension, but their positions are controlled by the `head_dim` and `token_dim` keyword arguments. All non-first-3 dimensions (batching dims) must match between `q`, `k`, and `v`.

**Keyword arguments**

* `head_dim::Int=1`: Which of the first three dimensions indexes the per-head embedding dimension.

* `token_dim::Int=3`: Which of the first three dimensions indexes tokens. `head_dim` and `token_dim` must be different and must be in `1:3`.

* `scale=nothing`: Scaling factor applied to the dot products. By default `scale = sqrt(size(q, head_dim))` and the implementation uses the inverse (i.e. divides by `sqrt(d_head)`). You may pass a number to override.

* `mask=nothing`: Optional boolean mask broadcastable to the attention logits. When provided it selects which key positions are allowed (true = keep).

* `is_causal::Union{Bool,Nothing}=nothing`: If `true` a causal mask is constructed automatically with `NNlib.make_causal_mask(...; dims=token_dim)`. `is_causal` and `mask` cannot both be specified.

* `bias=nothing`: Optional additive bias applied to attention logits before softmax.

* `fdrop=identity`: Function applied to the attention probabilities (e.g. a dropout function). It is applied after softmax and after mask/bias are applied.

**Extended Help**

**Mathematical Formulation**

$$\begin{align\*}
&\text{Let } q\_{d,h,l,b},\ k\_{d,h,s,b},\ v\_{e,h,s,b} \text{ denote the elements of the query, key, and value tensors,} \\
&\text{where:} \\
&\quad d: \text{embedding dimension (head\_dim)} \\
&\quad h: \text{head index} \\
&\quad l: \text{query token index} \\
&\quad s: \text{key/value token index} \\
&\quad e: \text{value embedding dimension} \\
&\quad b: \text{batch index} \\
\\
&\text{Scaling:} \\
&\quad \text{scale} = \frac{1}{\sqrt{d\_{\text{head}}}} \text{ (default, overridable)} \\
\\
&\text{Attention logits:} \\
&\quad \text{logits}*{s,l,h,b} = \text{scale} \cdot \sum*{d=1}^{d\_{\text{head}}} k\_{d,h,s,b} , q\_{d,h,l,b} + \text{bias}*{s,l,h,b} \\
&\quad \text{If mask is provided: logits}*{s,l,h,b} \rightarrow -\infty \text{ where mask is false} \\
\\
&\text{Attention weights:} \\
&\quad \alpha\_{s,l,h,b} = \text{softmax}*s(\text{logits}*{:,l,h,b}) \\
\\
&\text{Output:} \\
&\quad \text{output}*{e,l,h,b} = \sum*{s=1}^{S} v\_{e,h,s,b} , \alpha\_{s,l,h,b} \\
\end{align\*}$$

**Behavior notes**

* The implementation asserts `N ≥ 4` and enforces that `q` and `k` have the same size along `head_dim`, and that `k` and `v` have the same token length (along `token_dim`).

* Grouped KV: If `k`/`v` provide fewer KV groups than `q`'s number of heads, they are repeated along the non-head batching dimension so that the number of KV groups divides the number of heads. This supports e.g. shared keys for multiple heads.

* Masking and bias are combined using the internal `apply_attention_mask_bias` helper which produces numerically stable logits before softmax.

**Returns**

* A tuple `(output, attn_weights)` where
  * `output` contains the attended values with the same batch/head layout as `q`/`v` (see implementation for exact ordering), and

  * `attn_weights` contains the attention probabilities (softmax over the key/token dimension).

**Performance**

* CPU: The implementation relies on `batched_matmul` (which may use LoopVectorization when available) for good CPU performance.

* Reactant: Lowered to a `dot_general`-style op. It is highly recommended to use this function instead of writing out the sdpa implementation directly.

source

## Batched Operations {#Batched-Operations}

```julia
batched_matmul(x::AbstractMatrix, y::AbstractArray{yT,3}) where {yT}
batched_matmul(x::AbstractArray{xT,3}, y::AbstractMatrix) where {xT}
batched_matmul(
    x::AbstractArray{xT,N},
    y::AbstractArray{yT,N};
    lhs_contracting_dim::Int=2,
    rhs_contracting_dim::Int=1,
    lhs_batching_dims::Dims{M}=ntuple(Base.Fix2(+, 2), Val(N - 2)),
    rhs_batching_dims::Dims{M}=ntuple(Base.Fix2(+, 2), Val(N - 2)),
) where {xT,yT,N,M}
```

Computes the batched matrix multiplication of `x` and `y`. The following types are supported for `x` and `y`:

* `AbstractMatrix` and `AbstractArray{<:Number,3}`: `x` is treated as a tensor with batch size `1`.

* `AbstractArray{<:Number,3}` and `AbstractMatrix`: `y` is treated as a tensor with batch size `1`.

* `AbstractArray{<:Number,N}` and `AbstractArray{<:Number,N}`: This is the general form. This case supports the keyword arguments. Size of each corresponding batch dimension must be equal (or `1`).

**Keyword Arguments**

* `lhs_contracting_dim`: The dimension along which `x` is contracted. Defaults to `2`.

* `rhs_contracting_dim`: The dimension along which `y` is contracted. Defaults to `1`.

* `lhs_batching_dims`: The batching dimensions of `x`. Defaults to the last N - 2 dimensions.

* `rhs_batching_dims`: The batching dimensions of `y`. Defaults to the last N - 2 dimensions.

**Output Shape**

The output shape of result `z` is computed as follows:

* `size(z, 1)`: size of the non-contracting dimension of `x`.

* `size(z, 2)`: size of the non-contracting dimension of `y`.

* `size(z)[3:end]`: size of each of the resolved batching dimensions (i.e. `max(size(x, lhs_batching_dims[j]), size(y, rhs_batching_dims[j])))`) of `x` and `y`.

**Performance Considerations**

* **CPU**: `x` and `y` are converted to 3D tensors and then we attempt to use a custom implementation based on LoopVectorization. If this fails, we fall back to `NNlib.batched_mul`.

::: tip Load `LoopVectorization.jl` to get faster batched matrix multiplication

On CPUs loading LoopVectorization adds faster implementations of batched matrix multiplication.

:::

* **GPU**: `x` and `y` are converted to 3D tensors and then we call `NNlib.batched_mul`.

* **Reactant**: We directly lower this function to `stablehlo.dot_general`.

source

## Bias Activation {#Bias-Activation}

```julia
bias_activation(σ, x, bias)
```

Applies the activation function `σ` elementwise to the result of broadcasted addition of `x` and `bias` along the penultimate dimension. A vector `x` is treated as a matrix with a single last dimension.

**Arguments**

* `σ`: Activation function

* `x`: Input to be transformed

* `bias`: Bias to be added. Can be `nothing`.

See also [`bias_activation!!`](/api/NN_Primitives/LuxLib#LuxLib.API.bias_activation!!), [`fast_activation`](/api/NN_Primitives/LuxLib#LuxLib.API.fast_activation).

source

```julia
bias_activation!!(σ, x, bias)
```

Same as [`bias_activation`](/api/NN_Primitives/LuxLib#LuxLib.API.bias_activation) but might update `x` in-place if possible. Users should not rely on `x` being mutated, it is recommended to use it like `y = bias_activation!!(σ, x, bias)`. If `x` is updated in-place, `y` aliases `x`.

See also [`bias_activation`](/api/NN_Primitives/LuxLib#LuxLib.API.bias_activation), [`fast_activation!!`](/api/NN_Primitives/LuxLib#LuxLib.API.fast_activation!!).

source

## Convolutional Layers {#Convolutional-Layers}

```julia
fused_conv_bias_activation(σ::F, weight::AbstractArray, x::AbstractArray,
    b::Optional{<:AbstractVector}, cdims::ConvDims) where {F}
```

Computes `σ.(conv(x, weight, cdims) .+ b)` (`b` is not exactly broadcasted like this, rather it is reshaped and broadcasted to the penultimate dimension) with the best possible implementation available. This operation fuses operations into a single kernel if possible, and minimizes reallocations by reusing the output buffer for multiple operations.

**Arguments**

* `σ`: Activation function

* `weight`: Weight tensor

* `x`: Input tensor

* `b`: Bias tensor (can be `nothing`)

* `cdims`: `ConvDims` object

**Notes on implementation**

* For CUDA Arrays, this uses fused CUDNN kernels when the activation is `identity` or `relu`. For other activations, it tries to fuse the operations on the Julia side.

* If any of the inputs, don't support setindexing (aka immutable arrays) we fallback to the generic non-mutating implementation.

* Maximum memory reuse and operation fusion is guaranteed for ChainRules compatible AD backends or backends that support mutation. Backends like `Tracker` and `ReverseDiff` fallback to the generic implementation.

* For Mixed-Precision Inputs on GPU, we type promote the inputs to the highest precision, with a warning.

source

## Dropout {#Dropout}

```julia
alpha_dropout(rng::AbstractRNG, x, p, training)
alpha_dropout(rng::AbstractRNG, x, p, training, α, A, B)
```

Alpha Dropout: Dropout ensuring that the mean and variance of the output remains same as the input. For details see [Klambauer *et al.* \[8\]](/references#klambauer2017self). Use the second call signature to avoid recomputing the constants for a fixed dropout probability.

**Arguments**

* `rng`: Random number generator

* `x`: Input Array

* `p`: Probability of an element to be dropped out

* `training`: Set to `Val(true)` or `True()` if running in training mode. Can be set to `nothing` to automatically determine if the function is being called within an autodiff context\`

* `α`: `-1.7580993408473766`. Computed at limit x tends to infinity, `selu(x) = -λβ = α`

* `A`: Scaling factor for the mean

* `B`: Scaling factor for the variance

**Returns**

* Output Array after applying alpha dropout

* Updated state for the random number generator

source

```julia
dropout(rng::AbstractRNG, x, p, training, invp, dims)
dropout(rng::AbstractRNG, x, mask, p, training, update_mask::Union{Val, StaticBool},
    invp, dims)
```

Dropout: Simple Way to prevent Neural Networks for Overfitting. For details see [Srivastava *et al.* \[9\]](/references#srivastava2014dropout).

**Arguments**

* `rng`: Random number generator

* `x`: Input Array

* `mask`: Dropout Mask. If not used then it is constructed automatically

* `p`: Probability of an element to be dropped out

* `training`: Set to `Val(true)` or `True()` if running in training mode. Can be set to `nothing` to automatically determine if the function is being called within an autodiff context

* `update_mask`: If `Val(true)` or `True()` then the mask is generated and used. Else, the `mask` provided is directly used

* `invp`: Inverse multiplied to the mask. Calculated as `invp = 1 / (1 - p)`.

**Returns**

* Output Array after applying dropout

* Dropout Mask (if `training == false`, the returned value is meaningless)

* Updated state for the random number generator

source

## Fully Connected Layers {#Fully-Connected-Layers}

```julia
fused_dense_bias_activation(σ::F, weight::AbstractMatrix, x::AbstractMatrix,
    b::Optional{<:AbstractVector}) where {F}
```

Compute `σ.(weight * x .+ b)` with the best possible implementation available. Currently this implementation attempts to minimize reallocations by reusing the output buffer for multiple operations.

**Arguments**

* `σ`: Activation function

* `weight`: Weight matrix

* `x`: Input matrix

* `b`: Bias vector (can be `nothing`)

**Notes on implementation**

* If any of the inputs, don't support setindexing (aka immutable arrays) we fallback to the generic non-mutating implementation.

* Maximum memory reuse and operation fusion is guaranteed for ChainRules compatible AD backends or backends that support mutation. Backends like `Tracker` and `ReverseDiff` fallback to the generic implementation.

* For CUDA Arrays, this uses a special fused implementation via cuBLASLt.

* For small CPU Arrays, we use LoopVectorization.jl. On `x86_64` we use Octavian for medium sized matrices. This is overridden if special BLAS implementations are loaded (currently `MKL`, `AppleAccelerate`, and `BLISBLAS`).

::: tip Load `Octavian.jl`

Loading `Octavian.jl` enables a polyalgorithm that uses different backends based on the input sizes.

:::

source

## Normalization {#Normalization}

```julia
batchnorm(x, scale, bias, running_mean, running_var, training,
    σ=identity, momentum = 0.1f0, epsilon = eps(eltype(x)) ^ (5 // 7))
```

Batch Normalization. For details see [Ioffe and Szegedy \[10\]](/references#ioffe2015batch).

Batch Normalization computes the mean and variance for each $D\_1 \times ... \times D\_{N - 2} \times 1 \times D\_N$ input slice and normalises the input accordingly.

**Arguments**

* `x`: Input to be Normalized

* `scale`: Scale factor ($\gamma$) (can be `nothing`)

* `bias`: Bias factor ($\beta$) (can be `nothing`)

* `running_mean`: Running mean (can be `nothing`)

* `running_var`: Running variance (can be `nothing`)

* `training`: Set to `Val(true)` or `True()` if running in training mode. Can be set to `nothing` to automatically determine if the function is being called within an autodiff  context

* `σ`: Activation function (default: `identity`)

* `momentum`: Momentum for updating running mean and variance (default: `0.1f0`)

* `epsilon`: Value added to the denominator for numerical stability (default: `eps(eltype(x)) ^ (5 / 7)`)

**Returns**

Normalized Array of same size as `x`. And a Named Tuple containing the updated running mean and variance.

source

```julia
groupnorm(x, scale, bias, groups::Int, σ::F=identity,
    epsilon=eps(eltype(x)) ^ (5 // 7))
```

Group Normalization. For details see [Wu and He \[11\]](/references#wu2018group).

This op is similar to batch normalization, but statistics are shared across equally-sized groups of channels and not shared across batch dimension. Thus, group normalization does not depend on the batch composition and does not require maintaining internal state for storing statistics.

**Arguments**

* `x`: Input to be Normalized

* `scale`: Scale factor ($\gamma$) (can be `nothing`)

* `bias`: Bias factor ($\beta$) (can be `nothing`)

* `groups`: Number of groups

* `σ`: Activation function (default: `identity`)

* `epsilon`: Value added to the denominator for numerical stability (default: `eps(eltype(x)) ^ (5 / 7)`)

**Returns**

The normalized array is returned.

source

```julia
instancenorm(x, scale, bias, training, act, epsilon = eps(eltype(x)) ^ (5 // 7))
instancenorm(x, scale, bias, running_mean, running_var, training, act, momentum,
    epsilon = eps(eltype(x)) ^ (5 // 7))
```

Instance Normalization. For details see [Ulyanov *et al.* \[4\]](/references#ulyanov2016instance).

Instance Normalization computes the mean and variance for each $D\_1 \times ... \times D\_{N - 2} \times 1 \times 1$ input slice and normalises the input accordingly.

**Arguments**

* `x`: Input to be Normalized (must be atleast 3D)

* `scale`: Scale factor ($\gamma$) (can be `nothing`)

* `bias`: Bias factor ($\beta$) (can be `nothing`)

* `running_mean`: Running mean (can be `nothing`)

* `running_var`: Running variance (can be `nothing`)

* `training`: Set to `Val(true)` or `True()` if running in training mode. Can be set to `nothing` to automatically determine if the function is being called within an autodiff  context

* `σ`: Activation function (default: `identity`)

* `epsilon`: Value added to the denominator for numerical stability (default: `eps(eltype(x)) ^ (5 / 7)`)

* `momentum`: Momentum for updating running mean and variance (default: `0.1f0`)

**Returns**

Normalized Array of same size as `x`. And a Named Tuple containing the updated running mean and variance.

source

```julia
layernorm(x::AbstractArray{xT, N}, scale, bias, σ = identity, dims=1:(N - 1),
    epsilon = eps(eltype(x)) ^ (5 / 7)) where {xT, N}
```

Layer Normalization. For details see [Ba *et al.* \[12\]](/references#ba2016layer).

Given an input array $x$, this layer computes

$$y = \frac{x - \mathbb{E}\[x]}{\sqrt{Var\[x] + \epsilon}} \* \gamma + \beta$$

and applies the activation function `σ` elementwise to `y`.

**Arguments**

* `x`: Input to be Normalized

* `scale`: Scale factor ($\gamma$) (can be `nothing`)

* `bias`: Bias factor ($\beta$) (can be `nothing`)

* `σ`: Activation function (default: `identity`)

* `dims`: Dimensions along which the mean and std of `x` is computed. If `nothing` is passed, the dims are inferred based on the dimensions of scale and bias. For example, if `x` is `N` dimensional and `scale` and `bias` are `M` dimensional, then the dims will be `1:(N - M)`.

* `epsilon`: Value added to the denominator for numerical stability (default: `eps(eltype(x)) ^ (5 / 7)`)

**Returns**

Normalized Array of same size as `x`.

source

## Helper Functions {#Helper-Functions}

```julia
internal_operation_mode(xs::Tuple)
internal_operation_mode(x::AbstractArray)
```

Returns the internal operation mode for the given array(s). This is useful to define custom implementations using different backends like simple Julia broadcasting, Kernel Abstractions, Loop Vectorization, etc.

Currently supported modes are:

* `GenericBroadcastOp`: This is the fallback for most types. For the following types this is the preferred mode:
  * Arrays with `fast_scalar_indexing` set to `False`.

  * Static Arrays

  * ReverseDiff Arrays

  * Tracker Arrays

  * ForwardDiff.Dual Arrays

  * Complex Arrays

* `GPUBroadcastOp{dev}`: GPU Arrays where `dev` is obtained from `get_device_type(xs)`. This option dispatches should preferably use `KernelAbstractions` or specialized vendor dispatches.

* `LoopedArrayOp`: CPU arrays that can be optimized using SIMD Loops, ideally using `LoopVectorization.jl` or `Polyester.jl`.

source
