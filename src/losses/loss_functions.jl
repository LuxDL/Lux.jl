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

$$agg\left(-y\tilde * \log\left(y\hat + \epsilon\right) - (1 - y\tilde) * \log\left(1 - y\hat + \epsilon\right)\right)$$

  - If `logits` is `true` or `Val(true)`:

$$agg\left((1 - y\tilde) * y\hat - log\sigma(y\hat)\right)$$

The value of $y\tilde$ is computed using label smoothing. If `label_smoothing` is `nothing`,
then no label smoothing is applied. If `label_smoothing` is a real number $\in [0, 1]$,
then the value of $y\tilde$ is:

$$y\tilde = (1 - \alpha) * y + \alpha * 0.5$$

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

Return the [binary focal loss](https://arxiv.org/pdf/1708.02002.pdf). The model input,
$y\hat$, is expected to be normalized (i.e. [softmax](@ref Softmax) output).

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
    return loss.agg(sum(-y .* (1 .- ŷϵ) .^ γ .+ log.(ŷϵ); loss.dims))
end

@kwdef @concrete struct HingeLoss <: AbstractLossFunction
    agg = mean
end

function __unsafe_apply_loss(loss::HingeLoss, ŷ, y)
    T = promote_type(eltype(ŷ), eltype(y))
    return __fused_agg(loss.agg, Base.Fix1(max, T(0)), 1 .- y .* ŷ)
end

@kwdef @concrete struct HuberLoss <: AbstractLossFunction
    delta = 1
    agg = mean
end

function __unsafe_apply_loss(loss::HuberLoss, ŷ, y)
    T = promote_type(eltype(ŷ), eltype(y))
    return __fused_agg(loss.agg, Base.Fix2(__huber_metric, T(loss.delta)), abs.(ŷ .- y))
end

function __huber_metric(err::T1, δ::T2) where {T1, T2}
    T = promote_type(T1, T2)
    x = T(1 // 2)
    return ifelse(err < δ, err^2 * x, δ * (err - x * δ))
end

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

$$agg\left(\left| y\hat - y \right|\right)$$

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

$$agg\left(\left( y\hat - y \right)^2\right)$$

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

$$agg\left(\left( \log\left( y\hat + \epsilon \right) - \log\left( y + \epsilon \right) \right)^2\right)$$

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

@kwdef @concrete struct PoissonLoss <: AbstractLossFunction
    agg = mean
    epsilon = nothing
end

function __unsafe_apply_loss(loss::PoissonLoss, ŷ, y)
    T = promote_type(eltype(ŷ), eltype(y))
    ϵ = __get_epsilon(T, loss.epsilon)
    return loss.agg(ŷ .- xlogy.(y, ŷ .+ ϵ))
end

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

@kwdef @concrete struct SquaredHingeLoss <: AbstractLossFunction
    agg = mean
end

function __unsafe_apply_loss(loss::SquaredHingeLoss, ŷ, y)
    T = promote_type(eltype(ŷ), eltype(y))
    return __fused_agg(loss.agg, abs2 ∘ Base.Fix2(max, T(0)), 1 .- y .* ŷ)
end

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
