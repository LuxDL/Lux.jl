function reshaped_bias_dims(x::AbstractArray, bias::AbstractVector)
    return ntuple(i -> ifelse(i == ndims(x) - 1, length(bias), 1), ndims(x))
end

reshape_bias(::AbstractArray, ::Nothing) = nothing
reshape_bias(::AbstractVector, bias::Union{AbstractVector, StaticVector}) = bias
function reshape_bias(x::AbstractArray, bias::AbstractVector)
    return reshape(bias, reshaped_bias_dims(x, bias))
end
function reshape_bias(x::AbstractArray{<:Any, N}, bias::StaticVector) where {N}
    return SArray{Tuple{reshaed_bias_dims(x, bias)...}, eltype(bias), N, length(bias)}(bias.data)
end

## Needed for type stability
function CRC.rrule(::typeof(reshape_bias), x::AbstractArray{<:Number, N},
        bias::AbstractVector{<:Number}) where {N}
    bias_r = reshape_bias(x, bias)
    𝒫bias = CRC.ProjectTo(bias)
    return bias_r, Δ -> (∂∅, ∂∅, 𝒫bias(vec(Δ)))
end

∇bias_add(::Nothing, Δ::AbstractArray) = ∂∅
function ∇bias_add(b::AbstractArray{<:Number, N}, Δ::AbstractArray{<:Number, N}) where {N}
    return reduce_sum(b, Δ)
end
function ∇bias_add(b::AbstractVector{<:Number}, Δ::AbstractArray{<:Number})
    return vec(reduce_sum(reshape_bias(Δ, b), Δ))
end

reduce_sum(::Nothing, ::NoTangent) = ∂∅
function reduce_sum(x::AbstractArray, y::AbstractArray)
    z = similar(x, promote_type(eltype(x), eltype(y)))
    sum!(z, y)
    return z
end
