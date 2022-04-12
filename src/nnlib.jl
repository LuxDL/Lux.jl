# Matrix-Matrix & Matrix-Vector Multiplication
"""
    fast_matmul(A, B)
    fast_matmul!(C, A, B)

Dispatch to Octavian for CPU and CUBLAS for GPU
"""
fast_matmul

@inbounds Base.@pure function fast_matmul(A::AbstractMatrix{T1}, B::AbstractArray{T2,N}) where {T1,T2,N}
    return reshape(fast_matmul(A, reshape(B, size(B, 1), :)), :, size(B)[2:end]...)
end

@inbounds Base.@pure function fast_matmul(A::AbstractMatrix{T1}, B::AbstractMatrix{T2}) where {T1,T2}
    size(A, 2) != size(B, 1) && throw(DimensionMismatch("$(size(A, 2)) != $(size(B, 1)) for Matrix-Matrix Multiply"))
    return fast_matmul!(similar(A, promote_type(T1, T2), (size(A, 1), size(B, 2))), A, B)
end

@inbounds Base.@pure function fast_matmul(A::AbstractMatrix{T1}, b::AbstractVector{T2}) where {T1,T2}
    size(A, 2) != length(b) && throw(DimensionMismatch("$(size(A, 2)) != $(length(b)) for Matrix-Vector Multiply"))
    return fast_matmul!(similar(A, promote_type(T1, T2), size(A, 1)), A, b)
end

function fast_matmul!(C::AbstractVecOrMat, A::AbstractMatrix, B::AbstractVecOrMat)
    # Octavian can have unreliable speed sometimes
    # return matmul!(C, A, B)
    return mul!(C, A, B)
end

function fast_matmul!(
    C::CuVecOrMat,
    A::Union{<:CuMatrix{T1},<:Adjoint{T1,<:CuVecOrMat{T1}},<:Transpose{T1,<:CuVecOrMat{T1}}},
    B::Union{<:CuVecOrMat{T2},<:Adjoint{T2,<:CuVecOrMat{T2}},<:Transpose{T2,<:CuVecOrMat{T2}}},
) where {T1,T2}
    return mul!(C, A, B)
end

# Normalization Implementation
function get_stats!(
    ::Val{track_stats}, ::Val{active}, μ, σ², x::AbstractArray{T,N}, reduce_dims, momentum::T
) where {track_stats,active,T,N}
    if track_stats
        if active
            # Training
            μ_batch = mean(x; dims=reduce_dims)
            σ²_batch = std(x; mean=μ_batch, dims=reduce_dims)
            _update_stats!(reduce_dims, N, μ_batch, σ²_batch, momentum, μ, σ²)
            return μ_batch, σ²_batch
        else
            # Testing
            stats_shape = ntuple(i -> i == N - 1 ? size(x, N - 1) : 1, N)
            return reshape(μ, stats_shape), reshape(σ², stats_shape)
        end
    else
        # No Statistics Tracking
        μ = mean(x; dims=reduce_dims)
        return μ, std(x; mean=μ, dims=reduce_dims)
    end
end

function _update_stats!(reduce_dims, N, _μ, _σ², momentum, μ, σ²)
    μnew = vec(N == reduce_dims[end] ? _μ : mean(_μ; dims=N))
    σ²new = vec(N == reduce_dims[end] ? _σ² : mean(_σ²; dims=N))
    @. μ = (1 - momentum) * μ + momentum * μnew
    @. σ² = (1 - momentum) * σ² + momentum * σ²new
    return nothing
end

function norm_forward(
    l::AbstractNormalizationLayer{affine,track_stats},
    ps::NamedTuple,
    states::NamedTuple,
    x::AbstractArray{T,N},
    reduce_dims,
    affine_shape,
) where {T,N,affine,track_stats}
    μ, σ² = get_stats!(
        Val(track_stats),
        Val(states.training == :auto ? istraining() : states.training),
        states.μ,
        states.σ²,
        x,
        reduce_dims,
        l.momentum,
    )
    if affine
        γ = reshape(ps.γ, affine_shape)
        β = reshape(ps.β, affine_shape)
        return @. l.λ(γ * (x - μ) / sqrt(σ² + l.ϵ) + β)
    else
        return @. l.λ((x - μ) / sqrt(σ² + l.ϵ))
    end
end

Base.@pure function normalization_forward(
    l::AbstractNormalizationLayer{affine,track_stats},
    x::AbstractArray{T,N},
    xmean::Union{Nothing,AbstractArray{T,N}},
    xvar::Union{Nothing,AbstractArray{T,N}},
    scale::Union{Nothing,AbstractArray{T,N}},
    bias::Union{Nothing,AbstractArray{T,N}},
    activation::AT;
    training::Bool
) where {T,N,affine,track_stats,AT}
    reduce_dims = get_reduce_dims(l, x)
    if !training
        # Computing the mean and variance for the batch
        if !track_stats
            batchmean = mean(x, dims=reduce_dims)
            batchvar = var(x; mean=batchmean, dims=reduce_dims, corrected=false)
        else
            batchmean = xmean
            batchvar = xvar
        end
    else
        batchmean = mean(x, dims=reduce_dims)
        batchvar = var(x; mean=batchmean, dims=reduce_dims, corrected=false)

        if track_stats
            mometum = l.momentum
            m = T(prod(size(x, i) for i in reduce_dims))
            # Note that the @. after equals to is intentional to prevent mutation
            xmean = @. (1 - mometum) * xmean + mometum * batchmean
            xvar = @. (1 - mometum) * xvar + mometum * batchvar * (m / (m - 1))
        end
    end

    if affine
        if AT == typeof(identity)
            x_normalized = @. scale * (x - batchmean) / √(batchvar + l.ϵ) + bias
        else
            x_normalized = @. activation(scale * (x - batchmean) / √(batchvar + l.ϵ) + bias)
        end
    else
        if AT == typeof(identity)
            x_normalized = @. (x - batchmean) / √(batchvar + l.ϵ)
        else
            x_normalized = @. activation((x - batchmean) / √(batchvar + l.ϵ))
        end
    end

    # the mean and variance should not be used in any form other than storing
    # for future iterations
    return x_normalized, xmean, xvar
end

function get_stats!(
    ::Val{track_stats},
    ::Val{active},
    μ::AbstractArray{T,1},
    σ²::AbstractArray{T,1},
    x::AbstractArray{T,N},
    momentum::T,
    μ_batch::AbstractArray{T,N},
    σ²_batch::AbstractArray{T,N},
    σ²_batch_intermediate::AbstractArray{T,N},
    l::AbstractNormalizationLayer,
) where {track_stats,active,T,N}
    if track_stats
        if active
            # Training
            mean!(μ_batch, x)
            var!(σ²_batch, σ²_batch_intermediate, x, μ_batch)
            _update_stats!!(μ_batch, σ²_batch, momentum, μ, σ², l)
        else
            # Testing
            selectdim(μ_batch, N - 1, :) .= μ
            selectdim(σ²_batch, N - 1, :) .= σ²
        end
    else
        # No Statistics Tracking
        mean!(μ_batch, x)
        var!(σ²_batch, σ²_batch_intermediate, x, μ_batch)
    end
    return μ_batch, σ²_batch
end

function _update_stats!!(μnew, σ²new, momentum, μ, σ², ::BatchNorm)
    @. μ = (1 - momentum) * μ + momentum * μnew
    @. σ² = (1 - momentum) * σ² + momentum * σ²new
    return nothing
end

function norm_forward!(
    y::AbstractArray{T,N},
    μ::AbstractArray{T,N},
    σ²::AbstractArray{T,N},
    σ²_intermediate::AbstractArray{T,N},
    γ::Union{AbstractArray{T,N},Nothing},
    β::Union{AbstractArray{T,N},Nothing},
    l::AbstractNormalizationLayer{affine,track_stats},
    ps::NamedTuple,
    states::NamedTuple,
    x::AbstractArray{T,N},
) where {T,N,affine,track_stats}
    get_stats!(
        Val(track_stats),
        Val(states.training == :auto ? istraining() : states.training),
        states.μ,
        states.σ²,
        x,
        l.momentum,
        μ,
        σ²,
        σ²_intermediate,
        l,
    )
    if affine
        selectdim(γ, N - 1, :) .= ps.γ
        selectdim(β, N - 1, :) .= ps.β
        @. y = l.λ(γ * (x - μ) / sqrt(σ² + l.ϵ) + β)
    else
        @. y = l.λ((x - μ) / sqrt(σ² + l.ϵ))
    end
    return y
end

# Convolution
conv_wrapper(x, weight, cdims) = conv(x, weight, cdims)

function conv_wrapper(x::SubArray{T,N,<:CuArray}, weight, cdims) where {T,N}
    return conv(copy(x), weight, cdims)
end

function fast_conv_bias_act(
    x::SubArray{T,N,<:CuArray},
    w::AbstractArray{wT,N},
    cdims::ConvDims,
    b::AbstractArray{bT,N},
    λ=identity;
    kwargs...
) where {T, wT, bT, N}
    # NOTE: Without this we wont use CUDNN
    return fast_conv_bias_act(copy(x), w, cdims, b, λ, kwargs...)
end

function fast_conv_bias_act(
    x::AbstractArray{xT,N},
    w::AbstractArray{wT,N},
    cdims::ConvDims,
    b::AbstractArray{bT,N},
    λ=identity;
    kwargs...
) where {xT, wT, bT, N}
    y = similar(x, promote_type(xT, wT, bT), NNlib.output_size(cdims)..., NNlib.channels_out(cdims), size(x, N))
    return fast_conv_bias_act!(y, x, w, cdims, b, λ, kwargs...)
end

function fast_conv_bias_act!(
    y::AbstractArray{yT,N},
    x::AbstractArray{xT,N},
    w::AbstractArray{wT,N},
    cdims::ConvDims,
    b::AbstractArray{bT,N},
    λ::T=identity;
    kwargs...
) where {yT, xT, wT, bT, N, T}
    conv!(y, x, w, cdims)
    if T == typeof(identity)
        @. y += b
    else
        @. y = λ(y + b)
    end
    return y
end

# Dropout
_dropout_shape(s, ::Colon) = size(s)
_dropout_shape(s, dims) = tuple((i ∉ dims ? 1 : si for (i, si) ∈ enumerate(size(s)))...)

_dropout_kernel(y::T, p::T, q::T) where {T} = y > p ? inv(q) : 0

function dropout(rng::AbstractRNG, x, p; dims=:)
    y = _dropout_mask(rng, x, p, dims=dims)
    return dropout(rng, y, x, p; dims=dims), y
end

function dropout(rng::AbstractRNG, mask, x, p; dims=:)
    return x .* mask
end
