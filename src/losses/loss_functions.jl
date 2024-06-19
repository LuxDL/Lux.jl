# In this file, doctests which differ in the printed Float32 values won't fail
```@meta
DocTestFilters = r"[0-9\.]+f0"
```

function (loss::AbstractLossFunction)(ŷ, y)
    __check_sizes(ŷ, y)
    return __unsafe_apply_loss(loss, ŷ, y)
end

function __unsafe_apply_loss end

@doc doc"""
    BinaryCrossEntropyLoss(; agg = mean, epsilon = nothing,
        label_smoothing::Union{Nothing, Real}=nothing,
        logits::Union{Bool, Val}=Val(false))

Binary Cross Entropy Loss with optional label smoothing and fused logit computation.

Returns the binary cross entropy loss computed as:

  - If `logits` is either `false` or `Val(false)`:

$$\text{agg}\left(-\tilde{y} * \log\left(\hat{y} + \epsilon\right) - (1 - \tilde{y}) * \log\left(1 - \hat{y} + \epsilon\right)\right)$$

  - If `logits` is `true` or `Val(true)`:

$$\text{agg}\left((1 - \tilde{y}) * \hat{y} - log\sigma(\hat{y})\right)$$

The value of $\tilde{y}$ is computed using label smoothing. If `label_smoothing` is
`nothing`, then no label smoothing is applied. If `label_smoothing` is a real number
$\in [0, 1]$, then the value of $\tilde{y}$ is:

$$\tilde{y} = (1 - \alpha) * y + \alpha * 0.5$$

where $\alpha$ is the value of `label_smoothing`.

## Example

```jldoctest
julia> bce = BinaryCrossEntropyLoss();

julia> y_bin = Bool[1, 0, 1];

julia> y_model = Float32[2, -1, pi]
3-element Vector{Float32}:
  2.0
 -1.0
  3.1415927

julia> logitbce = BinaryCrossEntropyLoss(; logits=Val(true));

julia> logitbce(y_model, y_bin)
0.160832f0

julia> bce(sigmoid.(y_model), y_bin)
0.16083185f0

julia> bce_ls = BinaryCrossEntropyLoss(label_smoothing=0.1);

julia> bce_ls(sigmoid.(y_model), y_bin) > bce(sigmoid.(y_model), y_bin)
true

julia> logitbce_ls = BinaryCrossEntropyLoss(label_smoothing=0.1, logits=Val(true));

julia> logitbce_ls(y_model, y_bin) > logitbce(y_model, y_bin)
true
```

See also [`CrossEntropyLoss`](@ref).
"""
@concrete struct BinaryCrossEntropyLoss{logits, L <: Union{Nothing, Real}} <:
                 AbstractLossFunction
    label_smoothing::L
    agg
    epsilon
end

function BinaryCrossEntropyLoss(;
        agg=mean, epsilon=nothing, label_smoothing::Union{Nothing, Real}=nothing,
        logits::Union{Bool, Val}=Val(false))
    label_smoothing !== nothing && @argcheck 0 ≤ label_smoothing ≤ 1
    return BinaryCrossEntropyLoss{__unwrap_val(logits)}(label_smoothing, agg, epsilon)
end

for logits in (true, false)
    return_expr = logits ? :(return loss.agg((1 .- y_smooth) .* ŷ .- logsigmoid.(ŷ))) :
                  :(return loss.agg(-xlogy.(y_smooth, ŷ .+ ϵ) .-
                                    xlogy.(1 .- y_smooth, 1 .- ŷ .+ ϵ)))

    @eval function __unsafe_apply_loss(loss::BinaryCrossEntropyLoss{$(logits)}, ŷ, y)
        T = promote_type(eltype(ŷ), eltype(y))
        ϵ = __get_epsilon(T, loss.epsilon)
        y_smooth = __label_smoothing_binary(loss.label_smoothing, y, T)
        $(return_expr)
    end
end

@doc doc"""
    BinaryFocalLoss(; gamma = 2, agg = mean, epsilon = nothing)

Return the binary focal loss [1]. The model input, $\hat{y}$, is expected to be normalized
(i.e. [softmax](@ref Softmax) output).

For $\gamma = 0$ this is equivalent to [`BinaryCrossEntropyLoss`](@ref).

## Example

```jldoctest
julia> y = [0  1  0
            1  0  1];

julia> ŷ = [0.268941  0.5  0.268941
            0.731059  0.5  0.731059];

julia> BinaryFocalLoss()(ŷ, y) ≈ 0.0728675615927385
true

julia> BinaryFocalLoss(gamma=0)(ŷ, y) ≈ BinaryCrossEntropyLoss()(ŷ, y)
true
```

See also [`FocalLoss`](@ref) for multi-class focal loss.

## References

[1] Lin, Tsung-Yi, et al. "Focal loss for dense object detection." Proceedings of the IEEE
international conference on computer vision. 2017.
"""
@kwdef @concrete struct BinaryFocalLoss <: AbstractLossFunction
    gamma = 2
    agg = mean
    epsilon = nothing
end

@inline function __unsafe_apply_loss(loss::BinaryFocalLoss, ŷ, y)
    T = promote_type(eltype(ŷ), eltype(y))
    γ = loss.gamma isa Integer ? loss.gamma : T(loss.gamma)
    ϵ = __get_epsilon(T, loss.epsilon)
    ŷϵ = ŷ .+ ϵ
    p_t = y .* ŷϵ + (1 .- y) .* (1 .- ŷϵ)
    return __fused_agg(loss.agg, -, (1 .- p_t) .^ γ .* log.(p_t))
end

@doc doc"""
    CrossEntropyLoss(; agg=mean, epsilon=nothing, dims=1,
        label_smoothing::Union{Nothing, Real}=nothing)

Return the cross entropy loss which is used in multi-class classification tasks. The input,
$\hat{y}$, is expected to be normalized (i.e. `softmax` output) if `logits` is `false` or
`Val(false)`.

The loss is calculated as:

$$\text{agg}\left(-\sum \tilde{y} \log(\hat{y} + \epsilon)\right)$$

where $\epsilon$ is the smoothing factor.

## Example

```jldoctest
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

julia> CrossEntropyLoss()(y_model, y)
1.6076053f0

julia> 5 * 1.6076053f0 ≈ CrossEntropyLoss(; agg=sum)(y_model, y)
true

julia> CrossEntropyLoss(label_smoothing=0.15)(y_model, y)
1.5776052f0
```

See also [`BinaryCrossEntropyLoss`](@ref) for binary classification tasks.
"""
@concrete struct CrossEntropyLoss{logits, L <: Union{Nothing, Real}} <: AbstractLossFunction
    label_smoothing::L
    dims
    agg
    epsilon
end

function CrossEntropyLoss(;
        dims=1, agg=mean, epsilon=nothing, label_smoothing::Union{Nothing, Real}=nothing,
        logits::Union{Bool, Val}=Val(false))
    label_smoothing !== nothing && @argcheck 0 ≤ label_smoothing ≤ 1
    return CrossEntropyLoss{__unwrap_val(logits)}(label_smoothing, dims, agg, epsilon)
end

for logits in (true, false)
    return_expr = logits ?
                  :(return __fused_agg(
        loss.agg, -, sum(y_smooth .* logsoftmax(ŷ; loss.dims); loss.dims))) :
                  :(return __fused_agg(
        loss.agg, -, sum(xlogy.(y_smooth, ŷ .+ ϵ); loss.dims)))

    @eval function __unsafe_apply_loss(loss::CrossEntropyLoss{$(logits)}, ŷ, y)
        T = promote_type(eltype(ŷ), eltype(y))
        ϵ = __get_epsilon(T, loss.epsilon)
        y_smooth = __label_smoothing(loss.label_smoothing, y, T)
        $(return_expr)
    end
end

@doc doc"""
    DiceCoeffLoss(; smooth = true, agg = mean)

Return the Dice Coefficient loss [1] which is used in segmentation tasks. The dice
coefficient is similar to the F1_score. Loss calculated as:

$$agg\left(1 - \frac{2 \sum \tilde{y} \hat{y} + \alpha}{\sum \tilde{y}^2 + \sum \hat{y}^2 + \alpha}\right)$$

where $\alpha$ is the smoothing factor (`smooth`).

## Example

```jldoctest
julia> y_pred = [1.1, 2.1, 3.1];

julia> DiceCoeffLoss()(y_pred, 1:3)
0.000992391663909964

julia> 1 - DiceCoeffLoss()(y_pred, 1:3)
0.99900760833609
```

## References

[1] Milletari, Fausto, Nassir Navab, and Seyed-Ahmad Ahmadi. "V-net: Fully convolutional
neural networks for volumetric medical image segmentation." 2016 fourth international
conference on 3D vision (3DV). Ieee, 2016.
"""
@kwdef @concrete struct DiceCoeffLoss <: AbstractLossFunction
    smooth = true
    agg = mean
end

function __unsafe_apply_loss(loss::DiceCoeffLoss, ŷ, y)
    T = promote_type(eltype(ŷ), eltype(y))
    α = T(loss.smooth)

    yŷ = y .* ŷ
    dims = __get_dims(yŷ)

    num = T(2) .* sum(yŷ; dims) .+ α
    den = sum(abs2, ŷ; dims) .+ sum(abs2, y; dims) .+ α

    return loss.agg(true - num ./ den)
end

@doc doc"""
    FocalLoss(; gamma = 2, dims = 1, agg = mean, epsilon = nothing)

Return the focal loss [1] which can be used in classification tasks with highly imbalanced
classes. It down-weights well-classified examples and focuses on hard examples.
The input, $\hat{y}$, is expected to be normalized (i.e. `softmax` output).

The modulating factor $\gamma$, controls the down-weighting strength. For $\gamma = 0$ this
is equivalent to [`CrossEntropyLoss`](@ref).

## Example

```jldoctest
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

## References

[1] Lin, Tsung-Yi, et al. "Focal loss for dense object detection." Proceedings of the IEEE
international conference on computer vision. 2017.
"""
@kwdef @concrete struct FocalLoss <: AbstractLossFunction
    gamma = 2
    dims = 1
    agg = mean
    epsilon = nothing
end

@inline function __unsafe_apply_loss(loss::FocalLoss, ŷ, y)
    T = promote_type(eltype(ŷ), eltype(y))
    γ = loss.gamma isa Integer ? loss.gamma : T(loss.gamma)
    ϵ = __get_epsilon(T, loss.epsilon)
    ŷϵ = ŷ .+ ϵ
    return loss.agg(sum(-y .* (1 .- ŷϵ) .^ γ .* log.(ŷϵ); loss.dims))
end

@doc doc"""
    HingeLoss(; agg = mean)

Return the hinge loss loss given the prediction `ŷ` and true labels `y` (containing
1 or -1); calculated as:

$$\text{agg}\left(\max(0, 1 - y \hat{y})\right)$$

Usually used with classifiers like Support Vector Machines.

## Example

```jldoctest
julia> loss = HingeLoss();

julia> y_true = [1, -1, 1, 1];

julia> y_pred = [0.1, 0.3, 1, 1.5];

julia> loss(y_pred, y_true)
0.55

julia> loss(y_pred[1], y_true[1])
0.9

julia> loss(y_pred[2], y_true[2])
1.3

julia> loss(y_pred[3], y_true[3])
0.0
```

See also [`SquaredHingeLoss`](@ref).
"""
@kwdef @concrete struct HingeLoss <: AbstractLossFunction
    agg = mean
end

function __unsafe_apply_loss(loss::HingeLoss, ŷ, y)
    T = promote_type(eltype(ŷ), eltype(y))
    return __fused_agg(loss.agg, Base.Fix1(max, T(0)), 1 .- y .* ŷ)
end

@doc doc"""
    HuberLoss(; delta = 1, agg = mean)

Returns the Huber loss, calculated as:

$$L = \begin{cases}
    0.5 * |y - \hat{y}|^2 & \text{if } |y - \hat{y}| \leq \delta \\
    \delta * (|y - \hat{y}| - 0.5 * \delta) & \text{otherwise}
\end{cases}$$

where $\delta$ is the `delta` parameter.

## Example

```jldoctest
julia> y_model = [1.1, 2.1, 3.1];

julia> HuberLoss()(y_model, 1:3)
0.005000000000000009

julia> HuberLoss(delta=0.05)(y_model, 1:3)
0.003750000000000005
```
"""
@kwdef @concrete struct HuberLoss <: AbstractLossFunction
    delta = 1
    agg = mean
end

function __unsafe_apply_loss(loss::HuberLoss, ŷ, y)
    T = promote_type(eltype(ŷ), eltype(y))
    return __fused_agg(loss.agg, Base.Fix2(__huber_metric, T(loss.delta)), abs.(ŷ .- y))
end

function __huber_metric(err::T1, δ::T2) where {T1, T2}
    x = promote_type(T1, T2)(0.5)
    return ifelse(err < δ, err^2 * x, δ * (err - x * δ))
end

@doc doc"""
    KLDivergenceLoss(; dims = 1, agg = mean, epsilon = nothing, label_smoothing = nothing)

Return the Kullback-Leibler Divergence loss between the predicted distribution $\hat{y}$
and the true distribution $y$:

The KL divergence is a measure of how much one probability distribution is different from
the other. It is always non-negative, and zero only when both the distributions are equal.

For `epsilon` and `label_smoothing`, see [`CrossEntropyLoss`](@ref).

## Example

```jldoctest
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
"""
@concrete struct KLDivergenceLoss{C <: CrossEntropyLoss} <: AbstractLossFunction
    agg
    dims
    celoss::C
end

function KLDivergenceLoss(; dims=1, agg=mean, epsilon=nothing, label_smoothing=nothing)
    celoss = CrossEntropyLoss(; dims, agg, epsilon, label_smoothing)
    return KLDivergenceLoss(agg, dims, celoss)
end

function __unsafe_apply_loss(loss::KLDivergenceLoss, ŷ, y)
    cross_entropy = __unsafe_apply_loss(loss.celoss, ŷ, y)
    entropy = loss.agg(sum(xlogx, y; loss.dims))
    return entropy + cross_entropy
end

@doc doc"""
    MAELoss(; agg = mean)

Returns the loss corresponding to mean absolute error:

$$\text{agg}\left(\left| \hat{y} - y \right|\right)$$

## Example

```jldoctest
julia> loss = MAELoss();

julia> y_model = [1.1, 1.9, 3.1];

julia> loss(y_model, 1:3)
0.10000000000000009
```
"""
@kwdef @concrete struct MAELoss <: AbstractLossFunction
    agg = mean
end

const L1Loss = MAELoss

@inline __unsafe_apply_loss(loss::MAELoss, ŷ, y) = __fused_agg(loss.agg, abs, ŷ .- y)

@doc doc"""
    MSELoss(; agg = mean)

Returns the loss corresponding to mean squared error:

$$\text{agg}\left(\left( \hat{y} - y \right)^2\right)$$

## Example

```jldoctest
julia> loss = MSELoss();

julia> y_model = [1.1, 1.9, 3.1];

julia> loss(y_model, 1:3)
0.010000000000000018
```

See also [`MSELoss`](@ref).
"""
@kwdef @concrete struct MSELoss <: AbstractLossFunction
    agg = mean
end

const L2Loss = MSELoss

@inline __unsafe_apply_loss(loss::MSELoss, ŷ, y) = __fused_agg(loss.agg, abs2, ŷ .- y)

@doc doc"""
    MSLELoss(; agg = mean, epsilon = nothing)

Returns the loss corresponding to mean squared logarithmic error:

$$\text{agg}\left(\left( \log\left( \hat{y} + \epsilon \right) - \log\left( y + \epsilon \right) \right)^2\right)$$

`epsilon` is added to both `y` and `ŷ` to prevent taking the logarithm of zero. If `epsilon`
is `nothing`, then we set it to `eps(<type of y and ŷ>)`.

## Example

```jldoctest
julia> loss = MSLELoss();

julia> loss(Float32[1.1, 2.2, 3.3], 1:3)
0.009084041f0

julia> loss(Float32[0.9, 1.8, 2.7], 1:3)
0.011100831f0
```
"""
@kwdef @concrete struct MSLELoss <: AbstractLossFunction
    agg = mean
    epsilon = nothing
end

@inline function __unsafe_apply_loss(loss::MSLELoss, ŷ, y)
    T = promote_type(eltype(ŷ), eltype(y))
    ϵ = __get_epsilon(T, loss.epsilon)
    return __fused_agg(loss.agg, abs2 ∘ log, (ŷ .+ ϵ) ./ (y .+ ϵ))
end

@doc doc"""
    PoissonLoss(; agg = mean, epsilon = nothing)

Return how much the predicted distribution $\hat{y}$ diverges from the expected Poisson
distribution $y$, calculated as:

$$\text{agg}\left(\hat{y} - y * \log(\hat{y})\right)$$

## Example

```jldoctest
julia> y_model = [1, 3, 3];  # data should only take integral values

julia> PoissonLoss()(y_model, 1:3)
0.502312852219817
```
"""
@kwdef @concrete struct PoissonLoss <: AbstractLossFunction
    agg = mean
    epsilon = nothing
end

function __unsafe_apply_loss(loss::PoissonLoss, ŷ, y)
    T = promote_type(eltype(ŷ), eltype(y))
    ϵ = __get_epsilon(T, loss.epsilon)
    return loss.agg(ŷ .- xlogy.(y, ŷ .+ ϵ))
end

@doc doc"""
    SiameseContrastiveLoss(; margin = true, agg = mean)

Return the contrastive loss [1] which can be useful for training Siamese Networks. It is
given by:

$$\text{agg}\left((1 - y) \hat{y}^2 + y * \max(0, \text{margin} - \hat{y})^2\right)$$

Specify `margin` to set the baseline for distance at which pairs are dissimilar.

## Example

```jldoctest
julia> ŷ = [0.5, 1.5, 2.5];

julia> SiameseContrastiveLoss()(ŷ, 1:3)
-4.833333333333333

julia> SiameseContrastiveLoss(margin=2)(ŷ, 1:3)
-4.0
```

## References

[1] Hadsell, Raia, Sumit Chopra, and Yann LeCun. "Dimensionality reduction by learning an
invariant mapping." 2006 IEEE computer society conference on computer vision and pattern
recognition (CVPR'06). Vol. 2. IEEE, 2006.
"""
@concrete struct SiameseContrastiveLoss <: AbstractLossFunction
    margin
    agg
end

function SiameseContrastiveLoss(; margin::Real=true, agg=mean)
    @argcheck margin ≥ 0
    return SiameseContrastiveLoss(margin, agg)
end

@inline function __unsafe_apply_loss(loss::SiameseContrastiveLoss, ŷ, y)
    z = @. (1 - y) * ŷ^2 + y * max(0, loss.margin - ŷ)^2
    return loss.agg(z)
end

@doc doc"""
    SquaredHingeLoss(; agg = mean)

Return the squared hinge loss loss given the prediction `ŷ` and true labels `y` (containing
1 or -1); calculated as:

$$\text{agg}\left(\max(0, 1 - y \hat{y})^2\right)$$

Usually used with classifiers like Support Vector Machines.

## Example

```jldoctest
julia> loss = SquaredHingeLoss();

julia> y_true = [1, -1, 1, 1];

julia> y_pred = [0.1, 0.3, 1, 1.5];

julia> loss(y_pred, y_true)
0.625

julia> loss(y_pred[1], y_true[1]) ≈ 0.81
true

julia> loss(y_pred[2], y_true[2]) ≈ 1.69
true

julia> loss(y_pred[3], y_true[3])
0.0
```

See also [`HingeLoss`](@ref).
"""
@kwdef @concrete struct SquaredHingeLoss <: AbstractLossFunction
    agg = mean
end

function __unsafe_apply_loss(loss::SquaredHingeLoss, ŷ, y)
    T = promote_type(eltype(ŷ), eltype(y))
    return __fused_agg(loss.agg, abs2 ∘ Base.Fix2(max, T(0)), 1 .- y .* ŷ)
end

@doc doc"""
    TverskyLoss(; beta = 0.7, smooth = true, agg = mean)

Return the Tversky loss [1]. Used with imbalanced data to give more weight to false
negatives. Larger `beta` weigh recall more than precision (by placing more emphasis on
false negatives). Calculated as:

$$1 - \frac{\sum \left(y * \hat{y}\right) + \alpha}{\sum \left(y * \hat{y} + (1 - \beta) * (1 - y) * \hat{y} + \beta y * (1 - \hat{y})\right) + \alpha}$$

where $\alpha$ is the smoothing factor (`smooth`).

## References

[1] Salehi, Seyed Sadegh Mohseni, Deniz Erdogmus, and Ali Gholipour. "Tversky loss function
for image segmentation using 3D fully convolutional deep networks." International workshop
on machine learning in medical imaging. Cham: Springer International Publishing, 2017.
"""
@kwdef @concrete struct TverskyLoss <: AbstractLossFunction
    beta = 0.7
    smooth = true
    agg = mean
end

function __unsafe_apply_loss(loss::TverskyLoss, ŷ, y)
    T = promote_type(eltype(ŷ), eltype(y))
    β = T(loss.beta)
    α = T(loss.smooth)

    yŷ = y .* ŷ
    dims = __get_dims(yŷ)

    TP = sum(yŷ; dims)
    FP = sum((true .- y) .* ŷ; dims)
    FN = sum(y .* (true .- ŷ); dims)

    return loss.agg(1 - (TP + α) / (TP + α * FP + β * FN + α))
end

```@meta
DocTestFilters = nothing
```
