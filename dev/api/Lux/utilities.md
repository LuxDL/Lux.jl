---
url: /dev/api/Lux/utilities.md
---
# Utilities {#Utilities}

## Training API {#Training-API}

Helper Functions making it easier to train `Lux.jl` models.

Training is meant to be simple and provide extremely basic functionality. We provide basic building blocks which can be seamlessly composed to create complex training pipelines.

```julia
TrainState
```

Training State containing:

* `model`: `Lux` model.

* `parameters`: Trainable Variables of the `model`.

* `states`: Non-trainable Variables of the `model`.

* `optimizer`: Optimizer from `Optimisers.jl`.

* `optimizer_state`: Optimizer State.

* `step`: Number of updates of the parameters made.

Internal fields:

* `cache`: Cached values. Implementations are free to use this for whatever they want.

* `allocator_cache`: Used by GPUArrays compatible backends to cache memory allocations.

* `objective_function`: Objective function might be cached.

::: warning Warning

Constructing this object directly shouldn't be considered a stable API. Use the version with the Optimisers API.

:::

source

```julia
TrainState(model::Lux.AbstractLuxLayer, ps, st, optimizer::Optimisers.AbstractRule)
```

Constructor for [`TrainState`](/api/Lux/utilities#Lux.Training.TrainState).

**Arguments**

* `ps`: Parameters of the model.

* `st`: States of the model.

* `model`: `Lux` model.

* `optimizer`: Optimizer from `Optimisers.jl`.

**Returns**

[`TrainState`](/api/Lux/utilities#Lux.Training.TrainState) object.

source

```julia
compute_gradients(
    ad::AbstractADType, objective_function::Function, data, ts::TrainState;
    sync::Bool=false, compile_options::Union{Missing,Reactant.CompileOptions}=missing
)
```

Compute the gradients of the objective function wrt parameters stored in `ts`.

**Backends & AD Packages**

| Supported Backends           | Packages Needed  |
|:---------------------------- |:---------------- |
| `AutoZygote`                 | `Zygote.jl`      |
| `AutoReverseDiff(; compile)` | `ReverseDiff.jl` |
| `AutoTracker`                | `Tracker.jl`     |
| `AutoEnzyme`                 | `Enzyme.jl`      |
| `AutoForwardDiff`            |                  |
| `AutoMooncake`               | `Mooncake.jl`    |

**Arguments**

* `ad`: Backend (from [ADTypes.jl](https://github.com/SciML/ADTypes.jl)) used to compute the gradients.

* `objective_function`: Objective function. The function must take 4 inputs – model, parameters, states and data. The function must return 3 values – loss, updated\_state, and any computed statistics.

* `data`: Data used to compute the gradients.

* `ts`: Current Training State. See [`TrainState`](/api/Lux/utilities#Lux.Training.TrainState).

**Keyword Arguments**

* `sync`: If `true`, then the compiled reactant function is compiled with `sync=true`. Typically reactant functions are asynchronous, which means if used with profiling or for timing, the timing will be inaccurate. Setting `sync=true` will ensure that the function will finish execution before this function returns. This is only used for Reactant Backend.

* `compile_options`: Compile options for the reactant function. See `Reactant.CompileOptions` for more details. This is only used for Reactant Backend.

**Return**

A 4-Tuple containing:

* `grads`: Computed Gradients.

* `loss`: Loss from the objective function.

* `stats`: Any computed statistics from the objective function.

* `ts`: Updated Training State.

**Known Limitations**

* `AutoReverseDiff(; compile=true)` is not supported for Lux models with non-empty state `st`. Additionally the returned stats must be empty (`NamedTuple()`). We catch these issues in most cases and throw an error.

* AutoForwardDiff only works with parameters that are AbstractArrays (e.g. ps=ComponentVector(ps))

::: danger Aliased Gradients

`grads` returned by this function might be aliased by the implementation of the gradient backend. For example, if you cache the `grads` from step `i`, the new gradients returned in step `i + 1` might be aliased by the old gradients. If you want to prevent this, simply use `copy(grads)` or `deepcopy(grads)` to make a copy of the gradients.

:::

source

```julia
apply_gradients(ts::TrainState, grads)
```

Update the parameters stored in `ts` using the gradients `grads`.

**Arguments**

* `ts`: [`TrainState`](/api/Lux/utilities#Lux.Training.TrainState) object.

* `grads`: Gradients of the loss function wrt `ts.params`.

**Returns**

Updated [`TrainState`](/api/Lux/utilities#Lux.Training.TrainState) object.

source

```julia
apply_gradients!(ts::TrainState, grads)
```

Update the parameters stored in `ts` using the gradients `grads`. This is an inplace version of [`apply_gradients`](/api/Lux/utilities#Lux.Training.apply_gradients).

**Arguments**

* `ts`: [`TrainState`](/api/Lux/utilities#Lux.Training.TrainState) object.

* `grads`: Gradients of the loss function wrt `ts.params`.

**Returns**

Updated [`TrainState`](/api/Lux/utilities#Lux.Training.TrainState) object.

source

```julia
single_train_step(
    backend, obj_fn::F, data, ts::TrainState;
    return_gradients=True(), sync::Bool=false,
    compile_options::Union{Nothing,Reactant.CompileOptions}=missing,
)
```

Perform a single training step. Computes the gradients using [`compute_gradients`](/api/Lux/utilities#Lux.Training.compute_gradients) and updates the parameters using [`apply_gradients`](/api/Lux/utilities#Lux.Training.apply_gradients). All backends supported via [`compute_gradients`](/api/Lux/utilities#Lux.Training.compute_gradients) are supported here.

In most cases you should use [`single_train_step!`](/api/Lux/utilities#Lux.Training.single_train_step!) instead of this function.

**Keyword Arguments**

* `return_gradients`: If `True()`, the gradients are returned. If `False()`, the returned gradients are `nothing`. Defaults to `True()`. This is only used for Reactant Backend.

* `sync`: If `true`, then the compiled reactant function is compiled with `sync=true`. Typically reactant functions are asynchronous, which means if used with profiling or for timing, the timing will be inaccurate. Setting `sync=true` will ensure that the function will finish execution before this function returns. This is only used for Reactant Backend.

* `compile_options`: Compile options for the reactant function. See `Reactant.CompileOptions` for more details. This is only used for Reactant Backend.

**Return**

Returned values are the same as [`single_train_step!`](/api/Lux/utilities#Lux.Training.single_train_step!).

source

```julia
single_train_step!(
    backend, obj_fn::F, data, ts::TrainState;
    return_gradients=True(), sync::Bool=false,
    compile_options::Union{Nothing,Reactant.CompileOptions}=missing,
)
```

Perform a single training step. Computes the gradients using [`compute_gradients`](/api/Lux/utilities#Lux.Training.compute_gradients) and updates the parameters using [`apply_gradients!`](/api/Lux/utilities#Lux.Training.apply_gradients!). All backends supported via [`compute_gradients`](/api/Lux/utilities#Lux.Training.compute_gradients) are supported here.

**Keyword Arguments**

* `return_gradients`: If `True()`, the gradients are returned. If `False()`, the returned gradients are `nothing`. Defaults to `True()`. This is only used for Reactant Backend.

* `sync`: If `true`, then the compiled reactant function is compiled with `sync=true`. Typically reactant functions are asynchronous, which means if used with profiling or for timing, the timing will be inaccurate. Setting `sync=true` will ensure that the function will finish execution before this function returns. This is only used for Reactant Backend.

* `compile_options`: Compile options for the reactant function. See `Reactant.CompileOptions` for more details. This is only used for Reactant Backend.

**Return**

Returned values are the same as [`compute_gradients`](/api/Lux/utilities#Lux.Training.compute_gradients). Note that despite the `!`, only the parameters in `ts` are updated inplace. Users should be using the returned `ts` object for further training steps, else there is no caching and performance will be suboptimal (and absolutely terrible for backends like `AutoReactant`).

source

## Loss Functions {#Loss-Functions}

Loss Functions Objects take 2 forms of inputs:

1. `ŷ` and `y` where `ŷ` is the predicted output and `y` is the target output.

2. `model`, `ps`, `st`, `(x, y)` where `model` is the model, `ps` are the parameters, `st` are the states and `(x, y)` are the input and target pair. Then it returns the loss, updated states, and an empty named tuple. This makes them compatible with the [Training API](/api/Lux/utilities#Training-API).

::: warning Warning

When using ChainRules.jl compatible AD (like Zygote), we only compute the gradients wrt the inputs and drop any gradients wrt the targets.

:::

```julia
GenericLossFunction(loss_fn; agg = mean)
```

Takes any function `loss_fn` that maps 2 number inputs to a single number output. Additionally, array inputs are efficiently broadcasted and aggregated using `agg`.

```julia
julia> mseloss = GenericLossFunction((ŷ, y) -> abs2(ŷ - y));

julia> y_model = [1.1, 1.9, 3.1];

julia> mseloss(y_model, 1:3) ≈ 0.01
true
```

**Special Note**

This function takes any of the [`LossFunctions.jl`](https://juliaml.github.io/LossFunctions.jl/stable/) public functions into the Lux Losses API with efficient aggregation.

source

```julia
BinaryCrossEntropyLoss(; agg = mean, epsilon = nothing,
    label_smoothing::Union{Nothing, Real}=nothing,
    logits::Union{Bool, Val}=Val(false))
```

Binary Cross Entropy Loss with optional label smoothing and fused logit computation.

Returns the binary cross entropy loss computed as:

* If `logits` is either `false` or `Val(false)`:

  $$\text{agg}\left(-\tilde{y} \* \log\left(\hat{y} + \epsilon\right) - (1 - \tilde{y}) \* \log\left(1 - \hat{y} + \epsilon\right)\right)$$

* If `logits` is `true` or `Val(true)`:

  $$\text{agg}\left((1 - \tilde{y}) \* \hat{y} - \log\sigma(\hat{y})\right)$$

The value of $\tilde{y}$ is computed using label smoothing. If `label_smoothing` is `nothing`, then no label smoothing is applied. If `label_smoothing` is a real number $\in \[0, 1]$, then the value of $\tilde{y}$ is:

$$\tilde{y} = (1 - \alpha) \* y + \alpha \* 0.5$$

where $\alpha$ is the value of `label_smoothing`.

**Extended Help**

**Example**

```julia
julia> bce = BinaryCrossEntropyLoss();

julia> y_bin = Bool[1, 0, 1];

julia> y_model = Float32[2, -1, pi]
3-element Vector{Float32}:
  2.0
 -1.0
  3.1415927

julia> logitbce = BinaryCrossEntropyLoss(; logits=Val(true));

julia> logitbce(y_model, y_bin) ≈ 0.160832f0
true

julia> bce(sigmoid.(y_model), y_bin) ≈ 0.16083185f0
true

julia> bce_ls = BinaryCrossEntropyLoss(label_smoothing=0.1);

julia> bce_ls(sigmoid.(y_model), y_bin) > bce(sigmoid.(y_model), y_bin)
true

julia> logitbce_ls = BinaryCrossEntropyLoss(label_smoothing=0.1, logits=Val(true));

julia> logitbce_ls(y_model, y_bin) > logitbce(y_model, y_bin)
true
```

source

```julia
BinaryFocalLoss(; gamma = 2, agg = mean, epsilon = nothing)
```

Return the binary focal loss \[[5](/references#lin2017focal)]. The model input, $\hat{y}$, is expected to be normalized (i.e. softmax output).

For $\gamma = 0$ this is equivalent to [`BinaryCrossEntropyLoss`](/api/Lux/utilities#Lux.BinaryCrossEntropyLoss).

**Example**

```julia
julia> y = [0  1  0
            1  0  1];

julia> ŷ = [0.268941  0.5  0.268941
            0.731059  0.5  0.731059];

julia> BinaryFocalLoss()(ŷ, y) ≈ 0.0728675615927385
true

julia> BinaryFocalLoss(gamma=0)(ŷ, y) ≈ BinaryCrossEntropyLoss()(ŷ, y)
true
```

source

```julia
CrossEntropyLoss(;
    agg=mean, epsilon=nothing, dims=1, logits::Union{Bool, Val}=Val(false),
    label_smoothing::Union{Nothing, Real}=nothing
)
```

Return the cross entropy loss which is used in multi-class classification tasks. The input, $\hat{y}$, is expected to be normalized (i.e. `softmax` output) if `logits` is `false` or `Val(false)`.

The loss is calculated as:

$$\text{agg}\left(-\sum \tilde{y} \log(\hat{y} + \epsilon)\right)$$

where $\epsilon$ is added for numerical stability. The value of $\tilde{y}$ is computed using label smoothing. If `label_smoothing` is `nothing`, then no label smoothing is applied. If `label_smoothing` is a real number $\in \[0, 1]$, then the value of $\tilde{y}$ is calculated as:

$$\tilde{y} = (1 - \alpha) \* y + \alpha \* \text{size along dim}$$

where $\alpha$ is the value of `label_smoothing`.

**Extended Help**

**Example**

```julia
julia> y = [1  0  0  0  1
            0  1  0  1  0
            0  0  1  0  0]
3×5 Matrix{Int64}:
 1  0  0  0  1
 0  1  0  1  0
 0  0  1  0  0

julia> y_model = softmax(reshape(-7:7, 3, 5) .* 1f0)
3×5 Matrix{Float32}:
 0.0900306  0.0900306  0.0900306  0.0900306  0.0900306
 0.244728   0.244728   0.244728   0.244728   0.244728
 0.665241   0.665241   0.665241   0.665241   0.665241

julia> CrossEntropyLoss()(y_model, y) ≈ 1.6076053f0
true

julia> 5 * 1.6076053f0 ≈ CrossEntropyLoss(; agg=sum)(y_model, y)
true

julia> CrossEntropyLoss(label_smoothing=0.15)(y_model, y) ≈ 1.5776052f0
true
```

source

```julia
DiceCoeffLoss(; smooth = true, agg = mean)
```

Return the Dice Coefficient loss \[[6](/references#milletari2016v)] which is used in segmentation tasks. The dice coefficient is similar to the F1\_score. Loss calculated as:

$$\text{agg}\left(1 - \frac{2 \sum y \hat{y} + \alpha}{\sum y^2 + \sum \hat{y}^2 + \alpha}\right)$$

where $\alpha$ is the smoothing factor (`smooth`).

**Example**

```julia
julia> y_pred = [1.1, 2.1, 3.1];

julia> DiceCoeffLoss()(y_pred, 1:3)  ≈ 0.000992391663909964
true

julia> 1 - DiceCoeffLoss()(y_pred, 1:3)  ≈ 0.99900760833609
true

julia> DiceCoeffLoss()(reshape(y_pred, 3, 1), reshape(1:3, 3, 1)) ≈ 0.000992391663909964
true
```

source

```julia
FocalLoss(; gamma = 2, dims = 1, agg = mean, epsilon = nothing)
```

Return the focal loss \[[5](/references#lin2017focal)] which can be used in classification tasks with highly imbalanced classes. It down-weights well-classified examples and focuses on hard examples. The input, $\hat{y}$, is expected to be normalized (i.e. `softmax` output).

The modulating factor $\gamma$, controls the down-weighting strength. For $\gamma = 0$ this is equivalent to [`CrossEntropyLoss`](/api/Lux/utilities#Lux.CrossEntropyLoss).

**Example**

```julia
julia> y = [1  0  0  0  1
            0  1  0  1  0
            0  0  1  0  0]
3×5 Matrix{Int64}:
 1  0  0  0  1
 0  1  0  1  0
 0  0  1  0  0

julia> ŷ = softmax(reshape(-7:7, 3, 5) .* 1f0)
3×5 Matrix{Float32}:
 0.0900306  0.0900306  0.0900306  0.0900306  0.0900306
 0.244728   0.244728   0.244728   0.244728   0.244728
 0.665241   0.665241   0.665241   0.665241   0.665241

julia> FocalLoss()(ŷ, y) ≈ 1.1277556f0
true
```

source

```julia
HingeLoss(; agg = mean)
```

Return the hinge loss loss given the prediction `ŷ` and true labels `y` (containing 1 or -1); calculated as:

$$\text{agg}\left(\max(0, 1 - y \hat{y})\right)$$

Usually used with classifiers like Support Vector Machines.

**Example**

```julia
julia> loss = HingeLoss();

julia> y_true = [1, -1, 1, 1];

julia> y_pred = [0.1, 0.3, 1, 1.5];

julia> loss(y_pred, y_true) ≈ 0.55
true
```

source

```julia
HuberLoss(; delta = 1, agg = mean)
```

Returns the Huber loss, calculated as:

$$L = \begin{cases}
0.5 \* |y - \hat{y}|^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta \* (|y - \hat{y}| - 0.5 \* \delta) & \text{otherwise}
\end{cases}$$

where $\delta$ is the `delta` parameter.

**Example**

```julia
julia> y_model = [1.1, 2.1, 3.1];

julia> HuberLoss()(y_model, 1:3) ≈ 0.005000000000000009
true

julia> HuberLoss(delta=0.05)(y_model, 1:3) ≈ 0.003750000000000005
true
```

source

```julia
KLDivergenceLoss(; dims = 1, agg = mean, epsilon = nothing, label_smoothing = nothing)
```

Return the Kullback-Leibler Divergence loss between the predicted distribution $\hat{y}$ and the true distribution $y$:

The KL divergence is a measure of how much one probability distribution is different from the other. It is always non-negative, and zero only when both the distributions are equal.

For `epsilon` and `label_smoothing`, see [`CrossEntropyLoss`](/api/Lux/utilities#Lux.CrossEntropyLoss).

**Example**

```julia
julia> p1 = [1 0; 0 1]
2×2 Matrix{Int64}:
 1  0
 0  1

julia> p2 = fill(0.5, 2, 2)
2×2 Matrix{Float64}:
 0.5  0.5
 0.5  0.5

julia> KLDivergenceLoss()(p2, p1) ≈ log(2)
true

julia> KLDivergenceLoss(; agg=sum)(p2, p1) ≈ 2 * log(2)
true

julia> KLDivergenceLoss(; epsilon=0)(p2, p2)
0.0

julia> KLDivergenceLoss(; epsilon=0)(p1, p2)
Inf
```

source

```julia
MAELoss(; agg = mean)
```

Returns the loss corresponding to mean absolute error:

$$\text{agg}\left(\left| \hat{y} - y \right|\right)$$

**Example**

```julia
julia> loss = MAELoss();

julia> y_model = [1.1, 1.9, 3.1];

julia> loss(y_model, 1:3) ≈ 0.1
true
```

source

```julia
MSELoss(; agg = mean)
```

Returns the loss corresponding to mean squared error:

$$\text{agg}\left(\left( \hat{y} - y \right)^2\right)$$

**Example**

```julia
julia> loss = MSELoss();

julia> y_model = [1.1, 1.9, 3.1];

julia> loss(y_model, 1:3) ≈ 0.01
true
```

source

```julia
MSLELoss(; agg = mean, epsilon = nothing)
```

Returns the loss corresponding to mean squared logarithmic error:

$$\text{agg}\left(\left( \log\left( \hat{y} + \epsilon \right) - \log\left( y + \epsilon \right) \right)^2\right)$$

`epsilon` is added to both `y` and `ŷ` to prevent taking the logarithm of zero. If `epsilon` is `nothing`, then we set it to `eps(<type of y and ŷ>)`.

**Example**

```julia
julia> loss = MSLELoss();

julia> loss(Float32[1.1, 2.2, 3.3], 1:3) ≈ 0.009084041f0
true

julia> loss(Float32[0.9, 1.8, 2.7], 1:3) ≈ 0.011100831f0
true
```

source

```julia
PoissonLoss(; agg = mean, epsilon = nothing)
```

Return how much the predicted distribution $\hat{y}$ diverges from the expected Poisson distribution $y$, calculated as:

$$\text{agg}\left(\hat{y} - y \* \log(\hat{y})\right)$$

**Example**

```julia
julia> y_model = [1, 3, 3];  # data should only take integral values

julia> PoissonLoss()(y_model, 1:3) ≈ 0.502312852219817
true
```

source

```julia
SiameseContrastiveLoss(; margin = true, agg = mean)
```

Return the contrastive loss \[[7](/references#hadsell2006dimensionality)] which can be useful for training Siamese Networks. It is given by:

$$\text{agg}\left((1 - y) \hat{y}^2 + y \* \max(0, \text{margin} - \hat{y})^2\right)$$

Specify `margin` to set the baseline for distance at which pairs are dissimilar.

**Example**

```julia
julia> ŷ = [0.5, 1.5, 2.5];

julia> SiameseContrastiveLoss()(ŷ, 1:3) ≈ -4.833333333333333
true

julia> SiameseContrastiveLoss(margin=2)(ŷ, 1:3) ≈ -4.0
true
```

source

```julia
SquaredHingeLoss(; agg = mean)
```

Return the squared hinge loss loss given the prediction `ŷ` and true labels `y` (containing 1 or -1); calculated as:

$$\text{agg}\left(\max(0, 1 - y \hat{y})^2\right)$$

Usually used with classifiers like Support Vector Machines.

**Example**

```julia
julia> loss = SquaredHingeLoss();

julia> y_true = [1, -1, 1, 1];

julia> y_pred = [0.1, 0.3, 1, 1.5];

julia> loss(y_pred, y_true) ≈ 0.625
true
```

source

## LuxOps Module {#LuxOps-Module}

```julia
LuxOps
```

This module is a part of `Lux.jl`. It contains operations that are useful in DL context. Additionally certain operations here alias Base functions to behave more sensibly with GPUArrays.

source

```julia
eachslice(x, dims::Val)
```

Same as `Base.eachslice` but doesn't produce a `SubArray` for the slices if `x` is a GPUArray.

Additional dispatches for RNN helpers are also provided for `TimeLastIndex` and `BatchLastIndex`.

source

```julia
foldl_init(op, x)
foldl_init(op, x, init)
```

Exactly same as `foldl(op, x; init)` in the forward pass. But, gives gradients wrt `init` in the backward pass.

source

```julia
getproperty(x, ::Val{v})
getproperty(x, ::StaticSymbol{v})
```

Similar to `Base.getproperty` but requires a `Val` (or `Static.StaticSymbol`). Additionally, if `v` is not present in `x`, then `nothing` is returned.

source

```julia
xlogx(x::Number)
```

Return `x * log(x)` for `x ≥ 0`, handling `x == 0` by taking the limit from above, to get zero.

source

```julia
xlogy(x::Number, y::Number)
```

Return `x * log(y)` for `y > 0`, and zero when `x == 0`.

source

```julia
istraining(::Val{training})
istraining(::StaticBool)
istraining(::Bool)
istraining(st::NamedTuple)
```

Returns `true` if `training` is `true` or if `st` contains a `training` field with value `true`. Else returns `false`.

source

```julia
multigate(x::AbstractArray, ::Val{N})
```

Split up `x` into `N` equally sized chunks (along dimension `1`).

source

```julia
rsqrt(x)
```

Computes the reciprocal square root of `x`. For all backends except `Reactant.jl`, this falls back to `inv(sqrt(x))`.

source

## Recursive Operations {#Recursive-Operations}

```julia
recursive_map(f, x, args...)
```

Similar to `fmap(f, args...)` but with restricted support for the notion of "leaf" types. However, this allows for more efficient and type stable implementations of recursive operations.

::: warning Deprecation Warning

Starting Lux v1.3.0, this function is deprecated in favor of `Functors.fmap`. Functors v0.5 made significant strides towards improving the performance of `fmap` and hence this function has been deprecated. Users are encouraged to use `Functors.fmap` instead.

:::

**How this works?**

For the following types it directly defines recursion rules:

1. `AbstractArray`: If eltype is `isbitstype`, then `f` is applied to the array, else we recurse on the array.

2. `Tuple/NamedTuple`: We recurse on the values.

3. `Number/Val/Nothing`: We directly apply `f`.

4. For all other types, we recurse on the fields using `Functors.fmap`.

::: tip Note

In most cases, users should gravitate towards `Functors.fmap` if it is being used outside of hot loops. Even for other cases, it is always recommended to verify the correctness of this implementation for specific usecases.

:::

source

```julia
recursive_add!!(x, y)
```

Recursively add the leaves of two nested structures `x` and `y`. In Functor language, this is equivalent to doing `fmap(+, x, y)`, but this implementation uses type stable code for common cases.

Any leaves of `x` that are arrays and allow in-place addition will be modified in place.

::: warning Deprecation Warning

Starting Lux v1.3.0, this function is deprecated in favor of `Functors.fmap`. Functors v0.5 made significant strides towards improving the performance of `fmap` and hence this function has been deprecated. Users are encouraged to use `Functors.fmap` instead.

:::

source

```julia
recursive_copyto!(x, y)
```

Recursively copy the leaves of two nested structures `x` and `y`. In Functor language, this is equivalent to doing `fmap(copyto!, x, y)`, but this implementation uses type stable code for common cases. Note that any immutable leaf will lead to an error.

::: warning Deprecation Warning

Starting Lux v1.3.0, this function is deprecated in favor of `Functors.fmap`. Functors v0.5 made significant strides towards improving the performance of `fmap` and hence this function has been deprecated. Users are encouraged to use `Functors.fmap` instead.

:::

source

```julia
recursive_eltype(x, unwrap_ad_types = Val(false))
```

Recursively determine the element type of a nested structure `x`. This is equivalent to doing `fmap(Lux.Utils.eltype, x)`, but this implementation uses type stable code for common cases.

For ambiguous inputs like `nothing` and `Val` types we return `Bool` as the eltype.

If `unwrap_ad_types` is set to `Val(true)` then for tracing and operator overloading based ADs (ForwardDiff, ReverseDiff, Tracker), this function will return the eltype of the unwrapped value.

source

```julia
recursive_make_zero(x)
```

Recursively create a zero value for a nested structure `x`. This is equivalent to doing `fmap(zero, x)`, but this implementation uses type stable code for common cases.

See also [`Lux.recursive_make_zero!!`](/api/Lux/utilities#Lux.recursive_make_zero!!).

::: warning Deprecation Warning

Starting Lux v1.3.0, this function is deprecated in favor of `Functors.fmap`. Functors v0.5 made significant strides towards improving the performance of `fmap` and hence this function has been deprecated. Users are encouraged to use `Functors.fmap` instead.

:::

source

```julia
recursive_make_zero!!(x)
```

Recursively create a zero value for a nested structure `x`. Leaves that can be mutated with in-place zeroing will be modified in place.

See also [`Lux.recursive_make_zero`](/api/Lux/utilities#Lux.recursive_make_zero) for fully out-of-place version.

::: warning Deprecation Warning

Starting Lux v1.3.0, this function is deprecated in favor of `Functors.fmap`. Functors v0.5 made significant strides towards improving the performance of `fmap` and hence this function has been deprecated. Users are encouraged to use `Functors.fmap` instead.

:::

source

## Updating Floating Point Precision {#Updating-Floating-Point-Precision}

By default, Lux uses Float32 for all parameters and states. To update the precision simply pass the parameters / states / arrays into one of the following functions.

```julia
f16(m)
```

Converts the `eltype` of `m` *floating point* values to `Float16`. To avoid recursion into structs mark them with `Functors.@leaf`.

source

```julia
f32(m)
```

Converts the `eltype` of `m` *floating point* values to `Float32`. To avoid recursion into structs mark them with `Functors.@leaf`.

source

```julia
f64(m)
```

Converts the `eltype` of `m` *floating point* values to `Float64`. To avoid recursion into structs mark them with `Functors.@leaf`.

source

```julia
bf16(m)
```

Converts the `eltype` of `m` *floating point* values to `BFloat16`. To avoid recursion into structs mark them with `Functors.@leaf`.

::: warning Warning

`BFloat16s.jl` needs to be loaded before using this function.

:::

::: tip Support for `BFloat16`

Most Lux operations aren't optimized for `BFloat16` yet. Instead this is meant to be used together with `Reactant.@compile`.

:::

source

## Element Type Matching {#Element-Type-Matching}

```julia
match_eltype(layer, ps, st, args...)
```

Helper function to "maybe" (see below) match the element type of `args...` with the element type of the layer's parameters and states. This is useful for debugging purposes, to track down accidental type-promotions inside Lux layers.

**Extended Help**

**Controlling the Behavior via Preferences**

Behavior of this function is controlled via the  `eltype_mismatch_handling` preference. The following options are supported:

* `"none"`: This is the default behavior. In this case, this function is a no-op, i.e., it simply returns `args...`.

* `"warn"`: This option will issue a warning if the element type of `args...` does not match the element type of the layer's parameters and states. The warning will contain information about the layer and the element type mismatch.

* `"convert"`: This option is same as `"warn"`, but it will also convert the element type of `args...` to match the element type of the layer's parameters and states (for the cases listed below).

* `"error"`: Same as `"warn"`, but instead of issuing a warning, it will throw an error.

::: warning Warning

We print the warning for type-mismatch only once.

:::

**Element Type Conversions**

For `"convert"` only the following conversions are done:

| Element Type of parameters/states | Element Type of `args...` | Converted to |
|:--------------------------------- |:------------------------- |:------------ |
| `Float64`                         | `Integer`                 | `Float64`    |
| `Float32`                         | `Float64`                 | `Float32`    |
| `Float32`                         | `Integer`                 | `Float32`    |
| `Float16`                         | `Float64`                 | `Float16`    |
| `Float16`                         | `Float32`                 | `Float16`    |
| `Float16`                         | `Integer`                 | `Float16`    |

source

## Stateful Layer {#Stateful-Layer}

This layer have been moved to \[`LuxCore.jl`]. See the [documentation](/api/Building_Blocks/LuxCore#LuxCore.StatefulLuxLayerImpl.StatefulLuxLayer) for more details.

## Compact Layer {#Compact-Layer}

```julia
@compact(kw...) do x
    ...
    @return y # optional (but recommended for best performance)
end
@compact(kw...) do x, p
    ...
    @return y # optional (but recommended for best performance)
end
@compact(forward::Function; name=nothing, dispatch=nothing, parameters...)
```

Creates a layer by specifying some `parameters`, in the form of keywords, and (usually as a `do` block) a function for the forward pass. You may think of `@compact` as a specialized `let` block creating local variables that are trainable in Lux. Declared variable names may be used within the body of the `forward` function. Note that unlike typical Lux models, the forward function doesn't need to explicitly manage states.

Defining the version with `p` allows you to access the parameters in the forward pass. This is useful when using it with SciML tools which require passing in the parameters explicitly.

**Reserved Kwargs:**

1. `name`: The name of the layer.

2. `dispatch`: The constructed layer has the type `Lux.CompactLuxLayer{dispatch}` which can be used for custom dispatches.

::: tip Tip

Check the Lux tutorials for more examples of using `@compact`.

:::

If you are passing in kwargs by splatting them, they will be passed as is to the function body. This means if your splatted kwargs contain a lux layer that won't be registered in the CompactLuxLayer. Additionally all of the device functions treat these kwargs as leaves.

**Special Syntax**

* `@return`: This macro doesn't really exist, but is used to return a value from the `@compact` block. Without the presence of this macro, we need to rely on closures which can lead to performance penalties in the reverse pass.
  * Having statements after the last `@return` macro might lead to incorrect code.

  * Don't do things like `@return return x`. This will generate non-sensical code like `<new var> = return x`. Essentially, `@return <expr>` supports any expression, that can be assigned to a variable.

  * Since this macro doesn't "exist", it cannot be imported as `using Lux: @return`. Simply use it in code, and `@compact` will understand it.

* `@init_fn`: Provide a function that will be used to initialize the layer's parameters or state. See the docs of [`@init_fn`](/api/Lux/utilities#Lux.@init_fn) for more details.

* `@non_trainable`: Mark a value as non-trainable. This bypasses the regular checks and places the value into the state of the layer. See the docs of [`@non_trainable`](/api/Lux/utilities#Lux.@non_trainable) for more details.

**Extended Help**

**Examples**

Here is a linear model:

```julia
julia> using Lux, Random

julia> r = @compact(w=ones(Float32, 3)) do x
           @return w .* x
       end
@compact(
    w = 3-element Vector{Float32},
) do x
    return w .* x
end       # Total: 3 parameters,
          #        plus 0 states.

julia> ps, st = Lux.setup(Xoshiro(0), r);

julia> r(Float32[1, 2, 3], ps, st)  # x is set to [1, 1, 1].
(Float32[1.0, 2.0, 3.0], NamedTuple())
```

Here is a linear model with bias and activation:

```julia
julia> d_in = 5
5

julia> d_out = 3
3

julia> d = @compact(W=ones(Float32, d_out, d_in), b=zeros(Float32, d_out), act=relu) do x
           y = W * x
           @return act.(y .+ b)
       end
@compact(
    W = 3×5 Matrix{Float32},
    b = 3-element Vector{Float32},
    act = relu,
) do x
    y = W * x
    return act.(y .+ b)
end       # Total: 18 parameters,
          #        plus 1 states.

julia> ps, st = Lux.setup(Xoshiro(0), d);

julia> d(ones(Float32, 5, 2), ps, st)[1] # 3×2 Matrix as output.
3×2 Matrix{Float32}:
 5.0  5.0
 5.0  5.0
 5.0  5.0

julia> ps_dense = (; weight=ps.W, bias=ps.b);

julia> first(d(Float32[1, 2, 3, 4, 5], ps, st)) ≈
       first(Dense(d_in => d_out, relu)(Float32[1, 2, 3, 4, 5], ps_dense, NamedTuple())) # Equivalent to a dense layer
true
```

Finally, here is a simple MLP. We can train this model just like any Lux model:

```julia
julia> n_in = 1;

julia> n_out = 1;

julia> nlayers = 3;

julia> model = @compact(w1=Dense(n_in, 128),
           w2=[Dense(128, 128) for i in 1:nlayers], w3=Dense(128, n_out), act=relu) do x
           embed = act.(w1(x))
           for w in w2
               embed = act.(w(embed))
           end
           out = w3(embed)
           @return out
       end
@compact(
    w1 = Dense(1 => 128),                         # 256 parameters
    w2 = NamedTuple(
        (1-3) = Dense(128 => 128),                # 49_536 (16_512 x 3) parameters
    ),
    w3 = Dense(128 => 1),                         # 129 parameters
    act = relu,
) do x
    embed = act.(w1(x))
    for w = w2
        embed = act.(w(embed))
    end
    out = w3(embed)
    return out
end       # Total: 49_921 parameters,
          #        plus 1 states.

julia> ps, st = Lux.setup(Xoshiro(0), model);

julia> size(first(model(randn(Float32, n_in, 32), ps, st)))  # 1×32 Matrix as output.
(1, 32)

julia> using Optimisers, Zygote

julia> x_data = collect(-2.0f0:0.1f0:2.0f0)';

julia> y_data = 2 .* x_data .- x_data .^ 3;

julia> optim = Optimisers.setup(Adam(), ps);

julia> loss_initial = sum(abs2, first(model(x_data, ps, st)) .- y_data);

julia> for epoch in 1:1000
           loss, gs = Zygote.withgradient(
               ps -> sum(abs2, first(model(x_data, ps, st)) .- y_data), ps)
           Optimisers.update!(optim, ps, gs[1])
       end;

julia> loss_final = sum(abs2, first(model(x_data, ps, st)) .- y_data);

julia> loss_initial > loss_final
true
```

You may also specify a `name` for the model, which will be used instead of the default printout, which gives a verbatim representation of the code used to construct the model:

```julia
julia> model = @compact(w=rand(Float32, 3), name="Linear(3 => 1)") do x
           @return sum(w .* x)
       end
Linear(3 => 1)               # 3 parameters
```

This can be useful when using `@compact` to hierarchically construct complex models to be used inside a `Chain`.

::: tip Type Stability

If your input function `f` is type-stable but the generated model is not type stable, it should be treated as a bug. We will appreciate issues if you find such cases.

:::

::: warning Parameter Count

Array Parameter don't print the number of parameters on the side. However, they do account for the total number of parameters printed at the bottom.

:::

source

```julia
@init_fn(fn, kind::Symbol = :parameter)
```

Create an initializer function for a parameter or state to be used for in a Compact Lux Layer created using [`@compact`](/api/Lux/utilities#Lux.@compact).

**Arguments**

* `fn`: The function to be used for initializing the parameter or state. This only takes a single argument `rng`.

* `kind`: If set to `:parameter`, the initializer function will be used to initialize the parameters of the layer. If set to `:state`, the initializer function will be used to initialize the states of the layer.

**Examples**

```julia
julia> using Lux, Random

julia> r = @compact(w=@init_fn(rng->randn32(rng, 3, 2)),
           b=@init_fn(rng->randn32(rng, 3), :state)) do x
           @return w * x .+ b
       end;

julia> ps, st = Lux.setup(Xoshiro(0), r);

julia> size(ps.w)
(3, 2)

julia> size(st.b)
(3,)

julia> size(r([1, 2], ps, st)[1])
(3,)
```

source

```julia
@non_trainable(x)
```

Mark a value as non-trainable. This bypasses the regular checks and places the value into the state of the layer.

**Arguments**

* `x`: The value to be marked as non-trainable.

**Examples**

```julia
julia> using Lux, Random

julia> r = @compact(w=ones(3), w_fixed=@non_trainable(rand(3))) do x
           @return sum(w .* x .+ w_fixed)
       end;

julia> ps, st = Lux.setup(Xoshiro(0), r);

julia> size(ps.w)
(3,)

julia> size(st.w_fixed)
(3,)

julia> res, st_ = r([1, 2, 3], ps, st);

julia> st_.w_fixed == st.w_fixed
true

julia> res isa Number
true
```

source

## Miscellaneous {#Miscellaneous}

```julia
set_dispatch_doctor_preferences!(mode::String)
set_dispatch_doctor_preferences!(; luxcore::String="disable", luxlib::String="disable")
```

Set the dispatch doctor preference for `LuxCore` and `LuxLib` packages.

`mode` can be `"disable"`, `"warn"`, or `"error"`. For details on the different modes, see the [DispatchDoctor.jl](https://astroautomata.com/DispatchDoctor.jl/dev/) documentation.

If the preferences are already set, then no action is taken. Otherwise the preference is set. For changes to take effect, the Julia session must be restarted.

source
