# TODO: For the `dims === nothing` case, we can optimize using a loop vectorization and
#       kernel abstractions
function layernorm(x::AbstractArray{xT, N}, γ::Optional{<:AbstractArray},
        β::Optional{<:AbstractArray}, act::F, dims, epsilon::Real) where {N, F, xT}
    μ, σ² = mean_var(x; dims=compute_layernorm_dims(x, γ, β, dims), corrected=false)
    return affine_normalize(act, x, μ, σ², γ, β, epsilon)
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
