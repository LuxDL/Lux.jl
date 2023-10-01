using LuxCUDA
using .cuDNN: CUDNN_BN_MIN_EPSILON, cudnnBatchNormalizationBackward,
    cudnnBatchNormalizationForwardInference, CUDNN_BATCHNORM_SPATIAL,
    cudnnBatchNormalizationForwardTraining, cudnnTensorDescriptor, CUDNN_TENSOR_NCHW,
    cudnnDataType, dim4, scalingParameter, handle
import LuxLib: FP_32_64

# NOTE: This can be upstreamed to LuxCUDA once we drop support for v1.6
# Difference from the NNlib version: We expose the mean and inv_variance computed in the
# cudnn call, since they can be used at other places like forward mode AD

@inline function _wsize(x::AbstractArray{T, N}) where {T, N}
    return ntuple(i -> i == N - 1 ? size(x, N - 1) : 1, N)
end

function batchnorm_cudnn(γ::Nothing, β::Nothing, x::DenseCuArray, args...; kwargs...)
    affine_sz = _wsize(x)
    # Try to avoid hitting this in the first place. An easy workaround is to store the
    # gamma and bias parameters in states so that they are never trained
    g = fill!(similar(x, affine_sz), one(eltype(x)))
    b = fill!(similar(x, affine_sz), zero(eltype(x)))

    y, xμ, xσ⁻² = batchnorm_cudnn(g, b, x, args...; kwargs...)

    CUDA.unsafe_free!(g)
    CUDA.unsafe_free!(b)

    return y, xμ, xσ⁻²
end

function batchnorm_cudnn(g::DenseCuArray{T}, b::DenseCuArray{T}, x::DenseCuArray{T, 2},
    args...; kwargs...) where {T <: FP_32_64}
    x = reshape(x, 1, 1, size(x, 1), size(x, 2))
    y, xμ, xσ⁻² = batchnorm_cudnn(g, b, x, args...; kwargs...)
    return dropdims(y; dims=(1, 2)), xμ, xσ⁻²
end

function batchnorm_cudnn(g::DenseCuArray{T}, b::DenseCuArray{T},
    x::Union{DenseCuArray{T, 4}, DenseCuArray{T, 5}}, args...;
    kwargs...) where {T <: FP_32_64}
    return batchnorm_cudnn!(similar(x), g, b, x, args...; kwargs...)
end

function batchnorm_cudnn!(y::DenseCuArray{T}, g::DenseCuArray{T}, b::DenseCuArray{T},
    x::DenseCuArray{T}, running_μ, running_σ², momentum, ::Val{training};
    α=T(1), β=T(0), ϵ=T(1e-5)) where {T <: FP_32_64, training}
    dims = _wsize(x)
    if ϵ < CUDNN_BN_MIN_EPSILON
        @warn "eps $eps is too small for CuDNN, setting to CUDNN_BN_MIN_EPSILON=$CUDNN_BN_MIN_EPSILON"
        ϵ = CUDNN_BN_MIN_EPSILON
    end

    if running_μ === nothing || running_σ² === nothing
        running_μ !== running_σ² &&
            throw(ArgumentError("both or neither of running_μ and running_σ² must be nothing"))
        running_μ = CU_NULL
        running_σ² = CU_NULL
    end

    xd = cudnnTensorDescriptor(x)
    yd = cudnnTensorDescriptor(y)
    gd = cudnnTensorDescriptor(CUDNN_TENSOR_NCHW, cudnnDataType(T), Cint(length(dims)),
        dim4(dims, Val(CUDNN_TENSOR_NCHW)))

    if training
        mean = fill!(similar(x, dims), zero(T))
        ivar = fill!(similar(x, dims), one(T))

        cudnnBatchNormalizationForwardTraining(handle(), CUDNN_BATCHNORM_SPATIAL,
            scalingParameter(T, α), scalingParameter(T, β), xd, x, yd, y, gd, g, b,
            momentum, running_μ, running_σ², ϵ, mean, ivar)

        return y, mean, ivar
    else
        cudnnBatchNormalizationForwardInference(handle(), CUDNN_BATCHNORM_SPATIAL,
            scalingParameter(T, α), scalingParameter(T, β), xd, x, yd, y, gd, g, b,
            running_μ, running_σ², ϵ)

        return y, CU_NULL, CU_NULL
    end
end

function ∇batchnorm_cudnn(g::Nothing, b::Nothing, x::DenseCuArray, ∂y::DenseCuArray,
    running_μ, running_σ², args...; kwargs...)
    affine_sz = _wsize(x)
    g = fill!(similar(x, affine_sz), 1)
    b = fill!(similar(x, affine_sz), 0)

    ∂g, ∂b, ∂x = ∇batchnorm_cudnn(g, b, x, ∂y, running_μ, running_σ², args...; kwargs...)

    CUDA.unsafe_free!(g)
    CUDA.unsafe_free!(b)
    CUDA.unsafe_free!(∂g)
    CUDA.unsafe_free!(∂b)

    return (nothing, nothing, ∂x)
end

function ∇batchnorm_cudnn(g::DenseCuArray{T}, b::DenseCuArray{T}, x::DenseCuArray{T, 2},
    ∂y::DenseCuArray{T, 2}, running_μ, running_σ², args...; kwargs...) where {T <: FP_32_64}
    ∂g, ∂b, ∂x = ∇batchnorm_cudnn(g, b, reshape(x, 1, 1, size(x, 1), size(x, 2)),
        reshape(∂y, 1, 1, size(∂y, 1), size(∂y, 2)), running_μ, running_σ², args...;
        kwargs...)
    return (∂g, ∂b, dropdims(∂x; dims=(1, 2)))
end

function ∇batchnorm_cudnn(g::DenseCuArray{T}, b::DenseCuArray{T}, x::DenseCuArray{T},
    ∂y::DenseCuArray{T}, running_μ, running_σ², args...; kwargs...) where {T <: FP_32_64}
    ∂g = similar(g)
    ∂b = similar(b)
    ∂x = similar(x)
    cudnnBNBackward!(∂g, g, ∂b, ∂x, x, ∂y, running_μ, running_σ², args...; kwargs...)
    return (∂g, ∂b, ∂x)
end

function cudnnBNBackward!(∂g::DenseCuArray{T}, g::DenseCuArray{T}, ∂b::DenseCuArray{T},
    ∂x::DenseCuArray{T}, x::DenseCuArray{T}, ∂y::DenseCuArray{T}, running_μ, running_σ²,
    xmean, xivar; α=T(1), β=T(0), ϵ=T(1e-5), ∂α=T(1), ∂β=T(0)) where {T <: FP_32_64}
    if running_μ === nothing && running_σ² === nothing
        running_μ = CU_NULL
        running_σ² = CU_NULL
    end

    xd = cudnnTensorDescriptor(x)
    ∂yd = cudnnTensorDescriptor(∂y)
    ∂xd = cudnnTensorDescriptor(∂x)
    gd = cudnnTensorDescriptor(CUDNN_TENSOR_NCHW, cudnnDataType(T), Cint(length(_wsize(x))),
        dim4(_wsize(x), Val(CUDNN_TENSOR_NCHW)))

    xmean = xmean === nothing ? CU_NULL : xmean
    xivar = xivar === nothing ? CU_NULL : xivar

    if ϵ < CUDNN_BN_MIN_EPSILON
        @warn "eps $eps is too small for CuDNN, setting to CUDNN_BN_MIN_EPSILON=$CUDNN_BN_MIN_EPSILON"
        ϵ = CUDNN_BN_MIN_EPSILON
    end

    return cudnnBatchNormalizationBackward(handle(), CUDNN_BATCHNORM_SPATIAL,
        scalingParameter(T, α), scalingParameter(T, β), scalingParameter(T, ∂α),
        scalingParameter(T, ∂β), xd, x, ∂yd, ∂y, ∂xd, ∂x, gd, g, ∂g, ∂b, ϵ, xmean, xivar)
end
