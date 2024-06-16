# In this file, doctests which differ in the printed Float32 values won't fail
```@meta
DocTestFilters = r"[0-9\.]+f0"
```

function (loss::AbstractLossFunction)(ŷ, y)
    __check_sizes(ŷ, y)
    return __unsafe_apply_loss(loss, ŷ, y)
end

function __unsafe_apply_loss end

@kwdef @concrete struct L1Loss <: AbstractLossFunction
    agg = mean
end

@inline __unsafe_apply_loss(loss::L1Loss, ŷ, y) = __fused_agg(loss.agg, abs, ŷ .- y)

@kwdef @concrete struct MSELoss <: AbstractLossFunction
    agg = mean
end

@inline __unsafe_apply_loss(loss::MSELoss, ŷ, y) = __fused_agg(loss.agg, abs2, ŷ .- y)

@kwdef @concrete struct MSLELoss <: AbstractLossFunction
    agg = mean
    epsilon = nothing
end

@inline function __unsafe_apply_loss(loss::MSLELoss, ŷ, y)
    ϵ = loss.epsilon === nothing ? eps(eltype(ŷ)) : loss.epsilon
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
        ϵ = loss.epsilon === nothing ? eps(eltype(ŷ)) : loss.epsilon
        y_smooth = __label_smoothing(
            loss.label_smoothing, y, promote_type(eltype(ŷ), eltype(y)))
        $(return_expr)
    end
end

# TODO: HuberLoss
# TODO: BCELoss
# TODO: KLDivergenceLoss
# TODO: PoissonLoss
# TODO: HingeLoss
# TODO: SquaredHingeLoss
# TODO: DiceCoeffLoss
# TODO: TverskyLoss
# TODO: FocalLoss
# TODO: BinaryFocalLoss
# TODO: SimameseContrastiveLoss

```@meta
DocTestFilters = nothing
```
