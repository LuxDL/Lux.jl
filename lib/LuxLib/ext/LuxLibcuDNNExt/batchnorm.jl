# Difference from the NNlib version: We expose the mean and inv_variance computed in the
# cudnn call, since they can be used at other places like forward mode AD
function _wsize(x::AbstractArray{T, N}) where {T, N}
    return ntuple(i -> i == N - 1 ? size(x, N - 1) : 1, N)
end

function LuxLib.batchnorm_cudnn(γ::Nothing, β::Nothing, x::DenseCuArray, args...; kwargs...)
    affine_sz = _wsize(x)
    # Try to avoid hitting this in the first place. An easy workaround is to store the
    # gamma and bias parameters in states so that they are never trained
    g = fill!(similar(x, affine_sz), one(eltype(x)))
    b = fill!(similar(x, affine_sz), zero(eltype(x)))

    y, xμ, xσ⁻² = LuxLib.batchnorm_cudnn(g, b, x, args...; kwargs...)

    CUDA.unsafe_free!(g)
    CUDA.unsafe_free!(b)

    return y, xμ, xσ⁻²
end

function LuxLib.batchnorm_cudnn(
        g::DenseCuArray{T}, b::DenseCuArray{T}, x::DenseCuArray{T, 2},
        args...; kwargs...) where {T <: Union{Float32, Float64}}
    x = reshape(x, 1, 1, size(x, 1), size(x, 2))
    y, xμ, xσ⁻² = LuxLib.batchnorm_cudnn(g, b, x, args...; kwargs...)
    return dropdims(y; dims=(1, 2)), xμ, xσ⁻²
end

function LuxLib.batchnorm_cudnn(g::DenseCuArray{<:Union{Float32, Float64}},
        b::DenseCuArray{<:Union{Float32, Float64}},
        x::Union{DenseCuArray{<:Union{Float32, Float64}, 4},
            DenseCuArray{<:Union{Float32, Float64}, 5}},
        running_μ,
        running_σ²,
        args...;
        kwargs...)
    @warn "CUDNN batchnorm called with non-uniform eltypes. Promoting everything to the
        highest precision type. Avoid this code-path if possible." maxlog=1
    Tᵣₘ = running_μ === nothing ? Bool : eltype(running_μ)
    Tᵣᵥ = running_σ² === nothing ? Bool : eltype(running_σ²)
    T = promote_type(eltype(g), eltype(b), eltype(x), Tᵣₘ, Tᵣᵥ)

    ĝ = LuxLib._oftype_array(T, g)
    b̂ = LuxLib._oftype_array(T, b)
    x̂ = LuxLib._oftype_array(T, x)

    running_μ̂ = running_μ !== nothing ? LuxLib._oftype_array(T, running_μ) : running_μ
    running_σ̂² = running_σ² !== nothing ? LuxLib._oftype_array(T, running_σ²) : running_σ²

    y, xmean, xivar = LuxLib.batchnorm_cudnn(
        ĝ, b̂, x̂, running_μ̂, running_σ̂², args...; kwargs...)

    return (LuxLib._oftype_array(T, y), LuxLib._oftype_array(T, xmean),
        LuxLib._oftype_array(T, xivar))
end

function LuxLib.batchnorm_cudnn(g::DenseCuArray{T}, b::DenseCuArray{T},
        x::Union{DenseCuArray{T, 4}, DenseCuArray{T, 5}}, running_μ,
        running_σ², args...; kwargs...) where {T <: Union{Float32, Float64}}
    return batchnorm_cudnn!(similar(x), g, b, x, running_μ, running_σ², args...; kwargs...)
end

function batchnorm_cudnn!(y::DenseCuArray{T}, g::DenseCuArray{T}, b::DenseCuArray{T},
        x::DenseCuArray{T}, running_μ, running_σ², momentum, ::Val{training};
        α=T(1), β=T(0), ϵ=T(1e-5)) where {T <: Union{Float32, Float64}, training}
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
        cuDNN.dim4(dims, Val(CUDNN_TENSOR_NCHW)))

    if training
        mean = fill!(similar(x, dims), zero(T))
        ivar = fill!(similar(x, dims), one(T))

        cudnnBatchNormalizationForwardTraining(
            cuDNN.handle(), CUDNN_BATCHNORM_SPATIAL, cuDNN.scalingParameter(T, α),
            cuDNN.scalingParameter(T, β), xd, x, yd, y, gd, g,
            b, momentum, running_μ, running_σ², ϵ, mean, ivar)

        return y, mean, ivar
    else
        cudnnBatchNormalizationForwardInference(
            cuDNN.handle(), CUDNN_BATCHNORM_SPATIAL, cuDNN.scalingParameter(T, α),
            cuDNN.scalingParameter(T, β), xd, x, yd, y, gd, g, b, running_μ, running_σ², ϵ)

        return y, similar(x, zero.(dims)), similar(x, zero.(dims))
    end
end

function LuxLib.∇batchnorm_cudnn(g::Nothing, b::Nothing, x::DenseCuArray, ∂y::DenseCuArray,
        running_μ, running_σ², args...; kwargs...)
    affine_sz = _wsize(x)
    g = fill!(similar(x, affine_sz), 1)
    b = fill!(similar(x, affine_sz), 0)

    ∂g, ∂b, ∂x = LuxLib.∇batchnorm_cudnn(
        g, b, x, ∂y, running_μ, running_σ², args...; kwargs...)

    CUDA.unsafe_free!(g)
    CUDA.unsafe_free!(b)
    CUDA.unsafe_free!(∂g)
    CUDA.unsafe_free!(∂b)

    return nothing, nothing, ∂x
end

function LuxLib.∇batchnorm_cudnn(
        g::DenseCuArray{T}, b::DenseCuArray{T}, x::DenseCuArray{T, 2},
        ∂y::DenseCuArray{T, 2}, running_μ, running_σ², args...;
        kwargs...) where {T <: Union{Float32, Float64}}
    ∂g, ∂b, ∂x = LuxLib.∇batchnorm_cudnn(g, b, reshape(x, 1, 1, size(x, 1), size(x, 2)),
        reshape(∂y, 1, 1, size(∂y, 1), size(∂y, 2)),
        running_μ, running_σ², args...; kwargs...)
    return ∂g, ∂b, dropdims(∂x; dims=(1, 2))
end

function LuxLib.∇batchnorm_cudnn(g::DenseCuArray{<:Union{Float32, Float64}},
        b::DenseCuArray{<:Union{Float32, Float64}},
        x::DenseCuArray{<:Union{Float32, Float64}},
        ∂y::DenseCuArray{<:Union{Float32, Float64}},
        running_μ, running_σ², args...; kwargs...)
    @warn "CUDNN ∇batchnorm called with non-uniform eltypes. Promoting everything to the
        highest precision type. Avoid this code-path if possible." maxlog=1
    Tᵣₘ = running_μ === nothing ? Bool : eltype(running_μ)
    Tᵣᵥ = running_σ² === nothing ? Bool : eltype(running_σ²)
    T = promote_type(eltype(g), eltype(b), eltype(x), Tᵣₘ, Tᵣᵥ, eltype(∂y))

    ĝ = LuxLib._oftype_array(T, g)
    b̂ = LuxLib._oftype_array(T, b)
    x̂ = LuxLib._oftype_array(T, x)
    ∂ŷ = LuxLib._oftype_array(T, ∂y)
    running_μ̂ = running_μ !== nothing ? LuxLib._oftype_array(T, running_μ) : running_μ
    running_σ̂² = running_σ² !== nothing ? LuxLib._oftype_array(T, running_σ²) : running_σ²

    ∂g, ∂b, ∂x = LuxLib.∇batchnorm_cudnn(
        ĝ, b̂, x̂, ∂ŷ, running_μ̂, running_σ̂², args...; kwargs...)

    return (LuxLib._oftype_array(T, ∂g), LuxLib._oftype_array(T, ∂b),
        LuxLib._oftype_array(T, ∂x))
end

function LuxLib.∇batchnorm_cudnn(
        g::DenseCuArray{T}, b::DenseCuArray{T}, x::DenseCuArray{T}, ∂y::DenseCuArray{T},
        running_μ, running_σ², args...; kwargs...) where {T <: Union{Float32, Float64}}
    ∂g = similar(g)
    ∂b = similar(b)
    ∂x = similar(x)
    cudnnBNBackward!(∂g, g, ∂b, ∂x, x, ∂y, running_μ, running_σ², args...; kwargs...)
    return (∂g, ∂b, ∂x)
end

function cudnnBNBackward!(
        ∂g::DenseCuArray{T}, g::DenseCuArray{T}, ∂b::DenseCuArray{T}, ∂x::DenseCuArray{T},
        x::DenseCuArray{T}, ∂y::DenseCuArray{T}, running_μ, running_σ², xmean, xivar;
        α=T(1), β=T(0), ϵ=T(1e-5), ∂α=T(1), ∂β=T(0)) where {T <: Union{Float32, Float64}}
    if running_μ === nothing && running_σ² === nothing
        running_μ = CU_NULL
        running_σ² = CU_NULL
    end

    xd = cudnnTensorDescriptor(x)
    ∂yd = cudnnTensorDescriptor(∂y)
    ∂xd = cudnnTensorDescriptor(∂x)
    gd = cudnnTensorDescriptor(CUDNN_TENSOR_NCHW, cudnnDataType(T), Cint(length(_wsize(x))),
        cuDNN.dim4(_wsize(x), Val(CUDNN_TENSOR_NCHW)))

    xmean = xmean === nothing ? CU_NULL : xmean
    xivar = xivar === nothing ? CU_NULL : xivar

    if ϵ < CUDNN_BN_MIN_EPSILON
        @warn lazy"eps $eps is too small for CuDNN, setting to CUDNN_BN_MIN_EPSILON=$CUDNN_BN_MIN_EPSILON"
        ϵ = CUDNN_BN_MIN_EPSILON
    end

    return cudnnBatchNormalizationBackward(cuDNN.handle(), CUDNN_BATCHNORM_SPATIAL,
        cuDNN.scalingParameter(T, α), cuDNN.scalingParameter(T, β),
        cuDNN.scalingParameter(T, ∂α), cuDNN.scalingParameter(T, ∂β),
        xd, x, ∂yd, ∂y, ∂xd, ∂x, gd, g, ∂g, ∂b, ϵ, xmean, xivar)
end
