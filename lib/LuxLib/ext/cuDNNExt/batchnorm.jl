# Difference from the NNlib version: We expose the mean and inv_variance computed in the
# cudnn call, since they can be used at other places like forward mode AD
wsize(x::AbstractArray{T,N}, ::False) where {T,N} = (size(x, N - 1),)
function wsize(x::AbstractArray{T,N}, ::True) where {T,N}
    return ntuple(i -> i == N - 1 ? size(x, N - 1) : 1, N)
end

# Try to avoid hitting this in the first place. An easy workaround is to store the
# gamma and bias parameters in states so that they are never trained
function Impl.batchnorm_cudnn(::Nothing, ::Nothing, x::DenseCuArray, args...)
    affine_sz = wsize(x, False())
    γ = CUDA.ones(eltype(x), affine_sz)
    β = CUDA.zeros(eltype(x), affine_sz)

    y, xμ, xσ⁻² = Impl.batchnorm_cudnn(γ, β, x, args...)

    Utils.unsafe_free!(γ)
    Utils.unsafe_free!(β)

    return y, xμ, xσ⁻²
end

function Impl.batchnorm_cudnn(
    γ::DenseCuVector{T}, β::DenseCuVector{T}, x::DenseCuArray{T,2}, args...
) where {T<:cuDNNFloat}
    x = reshape(x, 1, 1, size(x, 1), size(x, 2))
    y, xμ, xσ⁻² = Impl.batchnorm_cudnn(γ, β, x, args...)
    return dropdims(y; dims=(1, 2)), xμ, xσ⁻²
end

function Impl.batchnorm_cudnn(
    γ::DenseCuVector{T},
    β::DenseCuVector{T},
    x::Union{DenseCuArray{T,4},DenseCuArray{T,5}},
    rμ::Optional{<:DenseCuVector{T}},
    rσ²::Optional{<:DenseCuVector{T}},
    args...,
) where {T<:cuDNNFloat}
    y = similar(x)
    μ, σ⁻² = batchnorm_cudnn!(y, γ, β, x, rμ, rσ², args...)
    return y, μ, σ⁻²
end

function batchnorm_cudnn!(
    y::DenseCuArray{T},
    γ′::DenseCuVector{T},
    β′::DenseCuVector{T},
    x::DenseCuArray{T},
    rμ′::Optional{<:DenseCuVector{T}},
    rσ²′::Optional{<:DenseCuVector{T}},
    m,
    ϵ,
    training::StaticBool,
) where {T<:cuDNNFloat}
    dims = wsize(x, True())

    γ = reshape(γ′, dims)
    β = reshape(β′, dims)
    rμ = safe_reshape(rμ′, dims...)
    rσ² = safe_reshape(rσ²′, dims...)

    (rμ === nothing) == (rσ² === nothing) ||
        throw(ArgumentError("both or neither of rμ and rσ² must be nothing"))

    if unsafe_known(training)
        return batchnorm_training!(
            y, x, γ, β; running_mean=rμ, running_var=rσ², momentum=m, epsilon=ϵ
        )
    else
        rμ === nothing && throw(ArgumentError("running statistics are required"))
        batchnorm_inference!(y, x, γ, β, rμ, rσ²; epsilon=ϵ)
        return similar(x, zero.(dims)), similar(x, zero.(dims))
    end
end

function Impl.∇batchnorm_cudnn(
    ::Nothing,
    ::Nothing,
    x::DenseCuArray,
    ∂y::DenseCuArray,
    rμ::Optional{<:DenseCuVector},
    rσ²::Optional{<:DenseCuVector},
    args...,
)
    affine_sz = wsize(x, False())
    γ = CUDA.ones(eltype(x), affine_sz)
    β = CUDA.zeros(eltype(x), affine_sz)

    ∂γ, ∂β, ∂x = Impl.∇batchnorm_cudnn(γ, β, x, ∂y, rμ, rσ², args...)

    Utils.unsafe_free!(γ)
    Utils.unsafe_free!(β)
    Utils.unsafe_free!(∂γ)
    Utils.unsafe_free!(∂β)

    return nothing, nothing, ∂x
end

function Impl.∇batchnorm_cudnn(
    γ::DenseCuVector{T},
    β::DenseCuVector{T},
    x::DenseCuArray{T,2},
    ∂y::DenseCuArray{T,2},
    rμ::Optional{<:DenseCuVector{T}},
    rσ²::Optional{<:DenseCuVector{T}},
    args...,
) where {T<:cuDNNFloat}
    ∂γ, ∂β, ∂x = Impl.∇batchnorm_cudnn(
        γ,
        β,
        reshape(x, 1, 1, size(x, 1), size(x, 2)),
        reshape(∂y, 1, 1, size(∂y, 1), size(∂y, 2)),
        rμ,
        rσ²,
        args...,
    )
    return ∂γ, ∂β, dropdims(∂x; dims=(1, 2))
end

function Impl.∇batchnorm_cudnn(
    γ::DenseCuVector{T},
    β::DenseCuVector{T},
    x::DenseCuArray{T,N},
    ∂y::DenseCuArray{T,N},
    rμ::Optional{<:DenseCuVector{T}},
    rσ²::Optional{<:DenseCuVector{T}},
    args...,
) where {T<:cuDNNFloat,N}
    ∂γ, ∂β, ∂x = similar(γ), similar(β), similar(x)
    ∇batchnorm_cudnn!(∂γ, γ, ∂β, ∂x, x, ∂y, rμ, rσ², args...)
    return ∂γ, ∂β, ∂x
end

function ∇batchnorm_cudnn!(
    ∂γ′::DenseCuVector{T},
    γ′::DenseCuVector{T},
    ∂β′::DenseCuVector{T},
    ∂x::DenseCuArray{T,N},
    x::DenseCuArray{T,N},
    ∂y::DenseCuArray{T,N},
    rμ′::Optional{<:DenseCuVector{T}},
    rσ²′::Optional{<:DenseCuVector{T}},
    xμ::Optional{<:DenseCuArray{<:cuDNNFloat,N}},
    xσ⁻²::Optional{<:DenseCuArray{<:cuDNNFloat,N}},
    ϵ,
) where {T<:cuDNNFloat,N}
    dims = wsize(x, True())

    ∂γ = reshape(∂γ′, dims)
    γ = reshape(γ′, dims)
    ∂β = reshape(∂β′, dims)

    if xμ === nothing || xσ⁻² === nothing
        xμ !== xσ⁻² &&
            throw(ArgumentError("both or neither of xμ and xσ⁻² must be nothing"))
        y = similar(x)
        β = CUDA.zeros(T, dims)
        xμ, xσ⁻² = batchnorm_training!(y, x, γ, β; epsilon=ϵ)
        Utils.unsafe_free!(y)
        Utils.unsafe_free!(β)
    end

    return batchnorm_gradient!(∂x, ∂γ, ∂β, ∂y, x, γ, xμ, xσ⁻²; epsilon=ϵ)
end
