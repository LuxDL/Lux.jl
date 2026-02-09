# In this file, doctests which differ in the printed Float32 values won't fail
```@meta
DocTestFilters = r"[0-9\.]+f0"
```
module LossFunctionImpl

using ArrayInterface: fast_scalar_indexing
using ChainRulesCore: ChainRulesCore, NoTangent, @non_differentiable, @thunk
using ForwardDiff: ForwardDiff, Dual, Partials
using Statistics: mean

using ..Utils: Utils
using ..LuxOps: xlogy

const CRC = ChainRulesCore

# Match the sizes of the inputs to the loss function
function check_sizes(ŷ::AbstractArray, y::AbstractArray)
    for d in 1:max(ndims(ŷ), ndims(y))
        if size(ŷ, d) != size(y, d)
            throw(
                DimensionMismatch("loss function expects size(ŷ) = $(size(ŷ)) to match \
                           size(y) = $(size(y))")
            )
        end
    end
    return nothing
end
check_sizes(_, __) = nothing

@non_differentiable check_sizes(::Any, ::Any)

# Aggregation. We are able to define custom aggregation fast paths
fused_agg(::typeof(mean), op::OP, x) where {OP} = fused_agg(sum, op, x) / length(x)

fused_agg(::typeof(sum), op::OP, x::Number) where {OP} = op(x)
fused_agg(::typeof(sum), op::OP, x) where {OP} = sum(op, x)

fused_agg(::typeof(mean), op::OP, x::Number, y::Number) where {OP} = op(x, y)
function fused_agg(::typeof(mean), op::OP, x::AbstractArray, y::AbstractArray) where {OP}
    return fused_agg(sum, op, x, y) / length(x)
end

fused_agg(::typeof(sum), op::OP, x::Number, y::Number) where {OP} = op(x, y)
function fused_agg(::typeof(sum), op::OP, x::AbstractArray, y::AbstractArray) where {OP}
    if fast_scalar_indexing(x) && fast_scalar_indexing(y)
        res = Core.Compiler.return_type(op, Tuple{eltype(x),eltype(y)})(0)
        @simd ivdep for i in eachindex(x, y)
            @inbounds res += op(x[i], y[i])
        end
        return res
    end
    return fallback_fused_agg(sum, op, x, y)
end

fused_agg(::Nothing, op::OP, args...) where {OP} = op.(args...)
fused_agg(f::F, op::OP, args...) where {F,OP} = fallback_fused_agg(f, op, args...)

@inline fallback_fused_agg(f::F, op::OP, args...) where {F,OP} = f(op.(args...))

function CRC.rrule(
    cfg::CRC.RuleConfig{>:CRC.HasReverseMode},
    ::typeof(fused_agg),
    ::typeof(sum),
    op::OP,
    x,
    y,
) where {OP}
    if has_custom_derivative(op)
        res = fused_agg(sum, op, x, y)
        ∇fused_agg_custom_derivative =
            Δ -> begin
                ∂x = @thunk derivative.(Ref(op), x, y) .* Δ
                return NoTangent(), NoTangent(), NoTangent(), ∂x, NoTangent()
            end
        return res, ∇fused_agg_custom_derivative
    end

    # Without custom derivatives use ForwardDiff for the looped implementation
    if fast_scalar_indexing(x) && fast_scalar_indexing(y)
        x_dual = Dual{Nothing,eltype(x),1}.(x, (Partials{1,eltype(x)}((one(eltype(x)),)),))
        x_partials = similar(x)
        T = eltype(x)
        res = Core.Compiler.return_type(op, Tuple{T,eltype(y)})(0)
        @inbounds @simd for i in eachindex(x_partials, x, y)
            x_dual = Dual{Nothing,T,1}(x[i], Partials{1,T}((one(T),)))
            tmp = op(x_dual, y[i])
            x_partials[i] = ForwardDiff.partials(tmp, 1)
            res += ForwardDiff.value(tmp)
        end
        ∇fused_agg_loop =
            Δ -> begin
                @simd ivdep for i in eachindex(x_partials)
                    @inbounds x_partials[i] *= Δ
                end
                return NoTangent(), NoTangent(), NoTangent(), x_partials, NoTangent()
            end
        return res, ∇fused_agg_loop
    end

    return CRC.rrule_via_ad(cfg, fallback_fused_agg, sum, op, x, y)
end

get_ϵ(::Type{T}, ϵ) where {T} = T(ϵ)
get_ϵ(::Type{T}, ::Nothing) where {T} = eps(float(T))

get_loss_dims(::AbstractVector) = Colon()
get_loss_dims(::AbstractArray{T,N}) where {T,N} = 1:(N - 1)

has_custom_derivative(::F) where {F} = false

has_custom_derivative(f::Utils.Fix3) = has_custom_derivative(f.f)
derivative(f::Utils.Fix3, x, y) = derivative(f.f, x, y, f.x)

# Functional forms of losses
l1_distance_loss(x::T1, y::T2) where {T1,T2} = abs(x - y)
has_custom_derivative(::typeof(l1_distance_loss)) = true
function derivative(::typeof(l1_distance_loss), x::T1, y::T2) where {T1,T2}
    return convert(T1, sign(x - y))
end

l2_distance_loss(x::T1, y::T2) where {T1,T2} = abs2(x - y)
has_custom_derivative(::typeof(l2_distance_loss)) = true
function derivative(::typeof(l2_distance_loss), x::T1, y::T2) where {T1,T2}
    return convert(T1, 2 * (x - y))
end

function huber_loss(x::T1, y::T2, δ::T3) where {T1,T2,T3}
    T = promote_type(T1, T2, T3)
    diff = x - y
    abs_diff = abs(diff)
    return ifelse(
        abs_diff ≤ δ, convert(T, 0.5) * abs2(diff), δ * (abs_diff - convert(T, 0.5) * δ)
    )
end
has_custom_derivative(::typeof(huber_loss)) = true
function derivative(::typeof(huber_loss), x::T, y::T2, δ::T3) where {T,T2,T3}
    diff = x - y
    return ifelse(abs(diff) ≤ δ, T(diff), T(δ) * convert(T, sign(diff)))
end

function l1_hinge_loss(x::T1, y::T2) where {T1,T2}
    agreement = x * y
    return max(oftype(agreement, false), true - agreement)
end
has_custom_derivative(::typeof(l1_hinge_loss)) = true
function derivative(::typeof(l1_hinge_loss), x::T1, y::T2) where {T1,T2}
    return T1(ifelse(x * y ≥ 1, false, true))
end

function l2_hinge_loss(x::T1, y::T2) where {T1,T2}
    agreement = x * y
    return ifelse(agreement ≥ 1, oftype(agreement, false), abs2(true - agreement))
end
has_custom_derivative(::typeof(l2_hinge_loss)) = true
function derivative(::typeof(l2_hinge_loss), x::T1, y::T2) where {T1,T2}
    agreement = x * y
    return T1(ifelse(agreement ≥ 1, false, 2 * (agreement - true)))
end

function siamese_contrastive_loss(x::T1, y::T2, margin=true) where {T1,T2}
    return (true - y) * x^2 + y * max(convert(promote_type(T1, T2), false), margin - x)^2
end

poisson_loss(x::T1, y::T2, ϵ) where {T1,T2} = x - xlogy(y, x + get_ϵ(T1, ϵ))

function msle_loss(x::T1, y::T2, ϵ) where {T1,T2}
    ϵ = get_ϵ(promote_type(T1, T2), ϵ)
    return log((x + ϵ) / (y + ϵ))^2
end

label_smoothing(::Nothing, y, ::Type{T}) where {T} = y
function label_smoothing(label_smoothing, y, ::Type{T}) where {T}
    label_smoothing = T(label_smoothing)
    return y .* (1 - label_smoothing) .+ label_smoothing ./ size(y, ndims(y) - 1)
end

label_smoothing_binary(::Nothing, y, ::Type{T}) where {T} = y
function label_smoothing_binary(label_smoothing, y, ::Type{T}) where {T}
    label_smoothing = T(label_smoothing)
    return y .* (1 - label_smoothing) .+ label_smoothing ./ 2
end

end

abstract type AbstractLossFunction <: Function end

function (loss::AbstractLossFunction)(model::AbstractLuxLayer, ps, st, (x, y))
    ŷ, stₙ = model(x, ps, st)
    return loss(ŷ, y), stₙ, (;)
end

function (loss::AbstractLossFunction)(ŷ, y)
    LossFunctionImpl.check_sizes(ŷ, y)
    return unsafe_apply_loss(loss, ŷ, y)
end

function unsafe_apply_loss end

@doc doc"""
    BinaryCrossEntropyLoss(; agg = mean, epsilon = nothing,
        label_smoothing::Union{Nothing, Real}=nothing,
        logits::Union{Bool, Val}=Val(false))

Binary Cross Entropy Loss with optional label smoothing and fused logit computation.

Returns the binary cross entropy loss computed as:

- If `logits` is either `false` or `Val(false)`:

  $$\text{agg}\left(-\tilde{y} * \log\left(\hat{y} + \epsilon\right) - (1 - \tilde{y}) * \log\left(1 - \hat{y} + \epsilon\right)\right)$$

- If `logits` is `true` or `Val(true)`:

  $$\text{agg}\left((1 - \tilde{y}) * \hat{y} - \log\sigma(\hat{y})\right)$$

The value of $\tilde{y}$ is computed using label smoothing. If `label_smoothing` is
`nothing`, then no label smoothing is applied. If `label_smoothing` is a real number
$\in [0, 1]$, then the value of $\tilde{y}$ is:

$$\tilde{y} = (1 - \alpha) * y + \alpha * 0.5$$

where $\alpha$ is the value of `label_smoothing`.

# Extended Help

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
"""
@concrete struct BinaryCrossEntropyLoss <: AbstractLossFunction
    logits <: Union{Val{true},Val{false}}
    label_smoothing <: Union{Nothing,Real}
    agg
    epsilon
end

function BinaryCrossEntropyLoss(;
    agg=mean,
    epsilon=nothing,
    label_smoothing::Union{Nothing,Real}=nothing,
    logits::Union{Bool,Val}=Val(false),
)
    label_smoothing !== nothing && @assert 0 ≤ label_smoothing ≤ 1
    logits isa Bool && (logits = Val(logits))
    return BinaryCrossEntropyLoss(logits, label_smoothing, agg, epsilon)
end

function unsafe_apply_loss(loss::BinaryCrossEntropyLoss, ŷ, y)
    T = promote_type(eltype(ŷ), eltype(y))
    ϵ = LossFunctionImpl.get_ϵ(T, loss.epsilon)
    ỹ = LossFunctionImpl.label_smoothing_binary(loss.label_smoothing, y, T)
    return compute_binary_cross_entropy(loss.logits, loss, ŷ, ỹ, ϵ)
end

function compute_binary_cross_entropy(::Val{true}, loss, ŷ, ỹ, ϵ)
    return loss.agg((1 .- ỹ) .* ŷ .- logsigmoid.(ŷ))
end

function compute_binary_cross_entropy(::Val{false}, loss, ŷ, ỹ, ϵ)
    return loss.agg(-xlogy.(ỹ, ŷ .+ ϵ) .- xlogy.(1 .- ỹ, 1 .- ŷ .+ ϵ))
end

@doc doc"""
    BinaryFocalLoss(; gamma = 2, agg = mean, epsilon = nothing)

Return the binary focal loss [lin2017focal](@cite). The model input, $\hat{y}$, is expected
to be normalized (i.e. softmax output).

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
"""
@kwdef @concrete struct BinaryFocalLoss <: AbstractLossFunction
    gamma = 2
    agg = mean
    epsilon = nothing
end

function unsafe_apply_loss(loss::BinaryFocalLoss, ŷ, y)
    T = promote_type(eltype(ŷ), eltype(y))
    γ = loss.gamma isa Integer ? loss.gamma : T(loss.gamma)
    ϵ = LossFunctionImpl.get_ϵ(T, loss.epsilon)
    ŷϵ = ŷ .+ ϵ
    p_t = y .* ŷϵ + (1 .- y) .* (1 .- ŷϵ)
    return LossFunctionImpl.fused_agg(loss.agg, -, (1 .- p_t) .^ γ .* log.(p_t))
end

@doc doc"""
    CrossEntropyLoss(;
        agg=mean, epsilon=nothing, dims=1, logits::Union{Bool, Val}=Val(false),
        label_smoothing::Union{Nothing, Real}=nothing
    )

Return the cross entropy loss which is used in multi-class classification tasks. The input,
$\hat{y}$, is expected to be normalized (i.e. `softmax` output) if `logits` is `false` or
`Val(false)`.

The loss is calculated as:

$$\text{agg}\left(-\sum \tilde{y} \log(\hat{y} + \epsilon)\right)$$

where $\epsilon$ is added for numerical stability. The value of $\tilde{y}$ is computed
using label smoothing. If `label_smoothing` is `nothing`, then no label smoothing is
applied. If `label_smoothing` is a real number $\in [0, 1]$, then the value of
$\tilde{y}$ is calculated as:

$$\tilde{y} = (1 - \alpha) * y + \alpha * \text{size along dim}$$

where $\alpha$ is the value of `label_smoothing`.

# Extended Help

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

julia> CrossEntropyLoss()(y_model, y) ≈ 1.6076053f0
true

julia> 5 * 1.6076053f0 ≈ CrossEntropyLoss(; agg=sum)(y_model, y)
true

julia> CrossEntropyLoss(label_smoothing=0.15)(y_model, y) ≈ 1.5776052f0
true
```
"""
@concrete struct CrossEntropyLoss <: AbstractLossFunction
    logits <: Union{Val{true},Val{false}}
    label_smoothing <: Union{Nothing,Real}
    dims
    agg
    epsilon
end

function CrossEntropyLoss(;
    dims=1,
    agg=mean,
    epsilon=nothing,
    label_smoothing::Union{Nothing,Real}=nothing,
    logits::Union{Bool,Val}=Val(false),
)
    label_smoothing !== nothing && @assert 0 ≤ label_smoothing ≤ 1
    logits isa Bool && (logits = Val(logits))
    return CrossEntropyLoss(logits, label_smoothing, dims, agg, epsilon)
end

function unsafe_apply_loss(loss::CrossEntropyLoss, ŷ, y)
    T = promote_type(eltype(ŷ), eltype(y))
    ϵ = LossFunctionImpl.get_ϵ(T, loss.epsilon)
    ỹ = LossFunctionImpl.label_smoothing(loss.label_smoothing, y, T)
    return compute_cross_entropy(loss.logits, loss, ŷ, ỹ, ϵ)
end

function compute_cross_entropy(::Val{true}, loss, ŷ, ỹ, ϵ)
    return LossFunctionImpl.fused_agg(
        loss.agg, -, sum(ỹ .* logsoftmax(ŷ; loss.dims); loss.dims)
    )
end

function compute_cross_entropy(::Val{false}, loss, ŷ, ỹ, ϵ)
    return LossFunctionImpl.fused_agg(loss.agg, -, sum(xlogy.(ỹ, ŷ .+ ϵ); loss.dims))
end

@doc doc"""
    DiceCoeffLoss(; smooth = true, agg = mean)

Return the Dice Coefficient loss [milletari2016v](@cite) which is used in segmentation
tasks. The dice coefficient is similar to the F1_score. Loss calculated as:

$$\text{agg}\left(1 - \frac{2 \sum y \hat{y} + \alpha}{\sum y^2 + \sum \hat{y}^2 + \alpha}\right)$$

where $\alpha$ is the smoothing factor (`smooth`).

## Example

```jldoctest
julia> y_pred = [1.1, 2.1, 3.1];

julia> DiceCoeffLoss()(y_pred, 1:3)  ≈ 0.000992391663909964
true

julia> 1 - DiceCoeffLoss()(y_pred, 1:3)  ≈ 0.99900760833609
true

julia> DiceCoeffLoss()(reshape(y_pred, 3, 1), reshape(1:3, 3, 1)) ≈ 0.000992391663909964
true
```
"""
@kwdef @concrete struct DiceCoeffLoss <: AbstractLossFunction
    smooth = true
    agg = mean
end

function unsafe_apply_loss(loss::DiceCoeffLoss, ŷ, y)
    T = promote_type(eltype(ŷ), eltype(y))
    α = T(loss.smooth)

    yŷ = y .* ŷ
    dims = LossFunctionImpl.get_loss_dims(yŷ)

    num = T(2) .* sum(yŷ; dims) .+ α
    den = sum(abs2, ŷ; dims) .+ sum(abs2, y; dims) .+ α

    return loss.agg(true .- num ./ den)
end

@doc doc"""
    FocalLoss(; gamma = 2, dims = 1, agg = mean, epsilon = nothing)

Return the focal loss [lin2017focal](@cite) which can be used in classification tasks with
highly imbalanced classes. It down-weights well-classified examples and focuses on hard
examples. The input, $\hat{y}$, is expected to be normalized (i.e. `softmax` output).

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
"""
@kwdef @concrete struct FocalLoss <: AbstractLossFunction
    gamma = 2
    dims = 1
    agg = mean
    epsilon = nothing
end

function unsafe_apply_loss(loss::FocalLoss, ŷ, y)
    T = promote_type(eltype(ŷ), eltype(y))
    γ = loss.gamma isa Integer ? loss.gamma : T(loss.gamma)
    ϵ = LossFunctionImpl.get_ϵ(T, loss.epsilon)
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

julia> loss(y_pred, y_true) ≈ 0.55
true
```
"""
HingeLoss(; agg=mean) = GenericLossFunction(LossFunctionImpl.l1_hinge_loss; agg)

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

julia> HuberLoss()(y_model, 1:3) ≈ 0.005000000000000009
true

julia> HuberLoss(delta=0.05)(y_model, 1:3) ≈ 0.003750000000000005
true
```
"""
function HuberLoss(; delta::Union{Nothing,AbstractFloat}=nothing, agg=mean)
    return GenericLossFunction(
        Utils.Fix3(LossFunctionImpl.huber_loss, ifelse(delta === nothing, true, delta)); agg
    )
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
@concrete struct KLDivergenceLoss <: AbstractLossFunction
    agg
    dims
    celoss <: CrossEntropyLoss
end

function KLDivergenceLoss(; dims=1, agg=mean, epsilon=nothing, label_smoothing=nothing)
    celoss = CrossEntropyLoss(; dims, agg, epsilon, label_smoothing)
    return KLDivergenceLoss(agg, dims, celoss)
end

function unsafe_apply_loss(loss::KLDivergenceLoss, ŷ, y)
    cross_entropy = unsafe_apply_loss(loss.celoss, ŷ, y)
    # Intentional broadcasting for Zygote type stability
    entropy = loss.agg(sum(xlogx.(y); loss.dims))
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

julia> loss(y_model, 1:3) ≈ 0.1
true
```
"""
MAELoss(; agg=mean) = GenericLossFunction(LossFunctionImpl.l1_distance_loss; agg)

const L1Loss = MAELoss

@doc doc"""
    MSELoss(; agg = mean)

Returns the loss corresponding to mean squared error:

$$\text{agg}\left(\left( \hat{y} - y \right)^2\right)$$

## Example

```jldoctest
julia> loss = MSELoss();

julia> y_model = [1.1, 1.9, 3.1];

julia> loss(y_model, 1:3) ≈ 0.01
true
```
"""
MSELoss(; agg=mean) = GenericLossFunction(LossFunctionImpl.l2_distance_loss; agg)

const L2Loss = MSELoss

@doc doc"""
    MSLELoss(; agg = mean, epsilon = nothing)

Returns the loss corresponding to mean squared logarithmic error:

$$\text{agg}\left(\left( \log\left( \hat{y} + \epsilon \right) - \log\left( y + \epsilon \right) \right)^2\right)$$

`epsilon` is added to both `y` and `ŷ` to prevent taking the logarithm of zero. If `epsilon`
is `nothing`, then we set it to `eps(<type of y and ŷ>)`.

## Example

```jldoctest
julia> loss = MSLELoss();

julia> loss(Float32[1.1, 2.2, 3.3], 1:3) ≈ 0.009084041f0
true

julia> loss(Float32[0.9, 1.8, 2.7], 1:3) ≈ 0.011100831f0
true
```
"""
function MSLELoss(; agg=mean, epsilon=nothing)
    return GenericLossFunction(Utils.Fix3(LossFunctionImpl.msle_loss, epsilon); agg)
end

@doc doc"""
    PoissonLoss(; agg = mean, epsilon = nothing)

Return how much the predicted distribution $\hat{y}$ diverges from the expected Poisson
distribution $y$, calculated as:

$$\text{agg}\left(\hat{y} - y * \log(\hat{y})\right)$$

## Example

```jldoctest
julia> y_model = [1, 3, 3];  # data should only take integral values

julia> PoissonLoss()(y_model, 1:3) ≈ 0.502312852219817
true
```
"""
function PoissonLoss(; agg=mean, epsilon=nothing)
    return GenericLossFunction(Utils.Fix3(LossFunctionImpl.poisson_loss, epsilon); agg)
end

@doc doc"""
    SiameseContrastiveLoss(; margin = true, agg = mean)

Return the contrastive loss [hadsell2006dimensionality](@cite) which can be useful for
training Siamese Networks. It is given by:

$$\text{agg}\left((1 - y) \hat{y}^2 + y * \max(0, \text{margin} - \hat{y})^2\right)$$

Specify `margin` to set the baseline for distance at which pairs are dissimilar.

## Example

```jldoctest
julia> ŷ = [0.5, 1.5, 2.5];

julia> SiameseContrastiveLoss()(ŷ, 1:3) ≈ -4.833333333333333
true

julia> SiameseContrastiveLoss(margin=2)(ŷ, 1:3) ≈ -4.0
true
```
"""
function SiameseContrastiveLoss(; margin=true, agg=mean)
    @assert margin ≥ 0
    return GenericLossFunction(
        Utils.Fix3(LossFunctionImpl.siamese_contrastive_loss, margin); agg
    )
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

julia> loss(y_pred, y_true) ≈ 0.625
true
```
"""
SquaredHingeLoss(; agg=mean) = GenericLossFunction(LossFunctionImpl.l2_hinge_loss; agg)

@doc doc"""
    GenericLossFunction(loss_fn; agg = mean)

Takes any function `loss_fn` that maps 2 number inputs to a single number output.
Additionally, array inputs are efficiently broadcasted and aggregated using `agg`.

```jldoctest
julia> mseloss = GenericLossFunction((ŷ, y) -> abs2(ŷ - y));

julia> y_model = [1.1, 1.9, 3.1];

julia> mseloss(y_model, 1:3) ≈ 0.01
true
```

## Special Note

This function takes any of the
[`LossFunctions.jl`](https://juliaml.github.io/LossFunctions.jl/stable/) public functions
into the Lux Losses API with efficient aggregation.
"""
@concrete struct GenericLossFunction <: AbstractLossFunction
    loss_fn
    agg
end

GenericLossFunction(loss_fn; agg=mean) = GenericLossFunction(loss_fn, agg)

function unsafe_apply_loss(loss::GenericLossFunction, ŷ, y)
    return LossFunctionImpl.fused_agg(loss.agg, loss.loss_fn, ŷ, y)
end

```@meta
DocTestFilters = nothing
```
