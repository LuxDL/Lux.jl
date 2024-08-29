function reshaped_bias_dims(x::AbstractArray, bias::AbstractVector)
    return ntuple(i -> ifelse(i == ndims(x) - 1, length(bias), 1), ndims(x))
end

reshape_bias(::AbstractArray, ::Nothing) = nothing
reshape_bias(::AbstractVector, bias::Union{AbstractVector, StaticVector}) = bias
function reshape_bias(x::AbstractArray, bias::AbstractVector)
    return reshape(bias, reshaped_bias_dims(x, bias))
end
function reshape_bias(x::AbstractArray{<:Any, N}, bias::StaticVector) where {N}
    return SArray{Tuple{reshaped_bias_dims(x, bias)...}, eltype(bias), N, length(bias)}(bias.data)
end

## Needed for type stability
function CRC.rrule(::typeof(reshape_bias), x::AbstractArray{xT, N},
        bias::AbstractVector{bT}) where {xT, bT, N}
    bias_r = reshape_bias(x, bias)
    𝒫bias = CRC.ProjectTo(bias)
    return bias_r, Δ -> (∂∅, ∂∅, 𝒫bias(vec(Δ)))
end

∇bias_add(::Nothing, Δ::AbstractArray) = ∂∅
function ∇bias_add(b::AbstractArray{xT, N}, Δ::AbstractArray{yT, N}) where {xT, yT, N}
    return reduce_sum(b, Δ)
end
function ∇bias_add(b::AbstractVector{xT}, Δ::AbstractArray{yT}) where {xT, yT}
    return vec(reduce_sum(reshape_bias(Δ, b), Δ))
end

reduce_sum(::Nothing, ::NoTangent) = ∂∅
function reduce_sum(x::AbstractArray, y::AbstractArray)
    z = similar(x, promote_type(eltype(x), eltype(y)))
    sum!(z, y)
    return z
end

function mean_var(x::AbstractArray; dims=:, corrected::Bool=true)
    μ = mean(x; dims)
    return μ, var(x; dims, corrected, mean=μ)
end

function CRC.rrule(::typeof(mean_var), x::AbstractArray; dims=:, corrected::Bool=true)
    μ, σ² = mean_var(x; dims, corrected)

    𝒫x = CRC.ProjectTo(x)
    ∇mean_var = @closure Δ -> begin
        ∂μ, ∂σ² = CRC.unthunk(Δ)
        n = dims_denom(x, dims)
        ∂x₁ = unsum(x, CRC.unthunk(∂μ) / n, dims)
        pre = 2 // (dims_denom(x, dims) - corrected)
        ∂x₂ = pre .* CRC.unthunk(∂σ²) .* (x .- μ)
        return NoTangent(), 𝒫x(add!!(∂x₁, ∂x₂))
    end

    return (μ, σ²), ∇mean_var
end

add!!(x, y) = add!!(is_mutable_array(x), x, y)
add!!(::True, x, y) = x .+= y
add!!(::False, x, y) = x .+ y

dims_denom(x, dims) = size(x, dims)
dims_denom(x, ::Colon) = length(x)
function dims_denom(x, dims::Union{Tuple, AbstractArray})
    return mapreduce(Base.Fix1(size, x), Base.mul_prod, unique(dims); init=1)
end

unsum(x, dy, _) = broadcast(last ∘ tuple, x, dy)
unsum(x, dy, ::Colon) = broadcast(last ∘ tuple, x, Ref(dy))
