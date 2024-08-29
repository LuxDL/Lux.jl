# TODO: For the `dims === nothing` case, we can optimize using a loop vectorization and
#       kernel abstractions
function layernorm(x::AbstractArray{xT, N}, γ::Optional{<:AbstractArray},
        β::Optional{<:AbstractArray}, act::F, dims, epsilon::Real) where {N, F, xT}
    μ, σ² = mean_var(x; dims=compute_layernorm_dims(x, γ, β, dims), corrected=false)
    γ′, β′ = expand_layernorm_dims(x, γ, β, dims)
    return affine_normalize(act, x, μ, σ², γ′, β′, epsilon)
end

function compute_layernorm_dims(::AbstractArray, ::Nothing, ::Nothing, ::Nothing)
    throw(ArgumentError("`dims` must be passed explicitly if `scale` and `bias` are \
                         `nothing`"))
end

function compute_layernorm_dims(::AbstractArray{xT, N}, ::AbstractArray{γT, M},
        ::AbstractArray{βT, M}, ::Nothing) where {xT, γT, βT, N, M}
    @assert N>M "`x` must have more dimensions than `scale` and `bias` when `dims` is \
                 `nothing`"
    return 1:(N - M)
end

function compute_layernorm_dims(
        ::AbstractArray, ::Optional{<:AbstractArray}, ::Optional{<:AbstractArray}, dims)
    return dims
end

CRC.@non_differentiable compute_layernorm_dims(::Any...)

expand_layernorm_dims(::AbstractArray, ::Nothing, ::Nothing, _) = nothing, nothing

function expand_layernorm_dims(::AbstractArray{xT, N}, γ::AbstractArray{γT, M},
        β::AbstractArray{βT, M}, ::Nothing) where {xT, γT, βT, N, M}
    new_γ_size = (size(γ)..., ntuple(i -> 1, N - M)...)
    new_β_size = (size(β)..., ntuple(i -> 1, N - M)...)
    return reshape(γ, new_γ_size), reshape(β, new_β_size)
end

function expand_layernorm_dims(::AbstractArray{yT, N}, γ::AbstractArray{γT, N},
        β::AbstractArray{βT, N}, dims) where {yT, γT, βT, N}
    return γ, β
end
