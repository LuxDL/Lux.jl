# In this file, doctests which differ in the printed Float32 values won't fail
```@meta
using Base: func_for_method_checked
DocTestFilters = r"[0-9\.]+f0"
```

function (loss::AbstractLossFunction)(ŷ, y)
    __check_sizes(ŷ, y)
    return __unsafe_apply_loss(loss, ŷ, y)
end

function __unsafe_apply_loss end

@kwdef @concrete struct MAELoss <: AbstractLossFunction
    agg = mean
end

const L1Loss = MAELoss

@inline __unsafe_apply_loss(loss::MAELoss, ŷ, y) = __fused_agg(loss.agg, abs, ŷ .- y)

@kwdef @concrete struct MSELoss <: AbstractLossFunction
    agg = mean
end

const L2Loss = MSELoss

@inline __unsafe_apply_loss(loss::MSELoss, ŷ, y) = __fused_agg(loss.agg, abs2, ŷ .- y)

@kwdef @concrete struct MSLELoss <: AbstractLossFunction
    agg = mean
    epsilon = nothing
end

@inline function __unsafe_apply_loss(loss::MSLELoss, ŷ, y)
    T = promote_type(eltype(ŷ), eltype(y))
    ϵ = __get_epsilon(T, loss.epsilon)
    return __fused_agg(loss.agg, abs2, log.((ŷ .+ ϵ) ./ (y .+ ϵ)))
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
    return_expr = logits ? :(return loss.agg((1 .- y_smooth) .* y̋ .- logsigmoid.(ŷ))) :
                  :(return loss.agg(-xlogy.(y_smooth, ŷ .+ ϵ) .-
                                    xlogy.(1 .- y_smooth, 1 .- ŷ .+ ϵ)))

    @eval function __unsafe_apply_loss(loss::BinaryCrossEntropyLoss{$(logits)}, ŷ, y)
        T = promote_type(eltype(ŷ), eltype(y))
        ϵ = __get_epsilon(T, loss.epsilon)
        y_smooth = __label_smoothing_binary(loss.label_smoothing, y, T)
        $(return_expr)
    end
end

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


# TODO: HuberLoss
# TODO: KLDivergenceLoss
# TODO: PoissonLoss
# TODO: HingeLoss
# TODO: SquaredHingeLoss

```@meta
DocTestFilters = nothing
```
