# Currently these don't do anything. But once we add LoopVectorization.jl and
# VectorizedStatistics.jl, we can will specialize the CPU dispatches to use them.
fast_mean(x::AbstractArray; dims=:) = fast_mean(internal_operation_mode(x), x; dims)
fast_mean(opmode, x::AbstractArray; dims=:) = mean(x; dims)

function fast_var(x::AbstractArray; mean=nothing, dims=:, corrected=true)
    fast_var(internal_operation_mode(x), x; mean, dims, corrected)
end
function fast_var(opmode, x::AbstractArray; mean=nothing, dims=:, corrected=true)
    return var(x; mean, dims, corrected)
end

function fast_mean_var(x::AbstractArray; dims=:, corrected=true)
    return fast_mean_var(internal_operation_mode(x), x; dims, corrected)
end

function fast_mean_var(opmode, x::AbstractArray; dims=:, corrected=true)
    μ = fast_mean(opmode, x; dims)
    σ² = fast_var(opmode, x; mean=μ, dims, corrected)
    return μ, σ²
end

function CRC.rrule(::typeof(fast_mean_var), x::AbstractArray; dims=:, corrected=true)
    opmode = internal_operation_mode(x)
    μ = fast_mean(opmode, x; dims)
    σ² = fast_var(opmode, x; mean=μ, dims, corrected)

    proj = CRC.ProjectTo(x)
    ∇fast_mean_var = @closure Δ -> begin
        ∂μ, ∂σ² = CRC.unthunk(Δ)
        n = _denom(x, dims)
        ∂x₁ = _unsum(x, CRC.unthunk(∂μ) / n, dims)
        pre = 2 // (_denom(x, dims) - corrected)
        ∂x₂ = pre .* CRC.unthunk(∂σ²) .* (x .- μ)
        ∂x = if can_setindex(∂x₁)
            @. ∂x₁ += ∂x₂
            ∂x₁
        else
            ∂x₁ .+ ∂x₂
        end
        return NoTangent(), proj(∂x)
    end

    return (μ, σ²), ∇fast_mean_var
end

_denom(x, dims) = size(x, dims)
_denom(x, ::Colon) = length(x)
function _denom(x, dims::Union{Tuple, AbstractArray})
    return mapreduce(Base.Fix1(size, x), Base.mul_prod, unique(dims); init=1)
end

_unsum(x, dy, dims) = broadcast(last ∘ tuple, x, dy)
_unsum(x, dy, ::Colon) = broadcast(last ∘ tuple, x, Ref(dy))
