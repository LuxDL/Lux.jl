# Matrix-Matrix & Matrix-Vector Multiplication
# @inbounds function fast_matmul(A::AbstractMatrix{T1}, B::AbstractArray{T2,N}) where {T1,T2,N}
#     return reshape(fast_matmul(A, reshape(B, size(B, 1), :)), :, size(B)[2:end]...)
# end

@inbounds function fast_matmul(A::AbstractMatrix{T1}, B::AbstractMatrix{T2}) where {T1,T2}
    # size(A, 2) != size(B, 1) && throw(DimensionMismatch("$(size(A, 2)) != $(size(B, 1)) for Matrix-Matrix Multiply"))
    # return fast_matmul!(similar(A, promote_type(T1, T2), (size(A, 1), size(B, 2))), A, B)
    return A * B
end

@inbounds function fast_matmul(A::AbstractMatrix{T1}, b::AbstractVector{T2}) where {T1,T2}
    # size(A, 2) != length(b) && throw(DimensionMismatch("$(size(A, 2)) != $(length(b)) for Matrix-Vector Multiply"))
    # return fast_matmul!(similar(A, promote_type(T1, T2), size(A, 1)), A, b)
    return A * b
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
function update_statistics(::AbstractNormalizationLayer, xmean, xvar, batchmean, batchvar, momentum, m)
    batchmean = mean(batchmean, dims=ndims(batchmean))
    batchvar = mean(batchvar, dims=ndims(batchvar))
    _xmean = @. (1 - momentum) * xmean + momentum * batchmean
    _xvar = @. (1 - momentum) * xvar + momentum * batchvar * (m / (m - 1))
    return (_xmean, _xvar)
end

function update_statistics(::BatchNorm, xmean, xvar, batchmean, batchvar, momentum, m)
    _xmean = @. (1 - momentum) * xmean + momentum * batchmean
    _xvar = @. (1 - momentum) * xvar + momentum * batchvar * (m / (m - 1))
    return (_xmean, _xvar)
end

function normalization_forward(
    l::AbstractNormalizationLayer{affine,track_stats},
    x::AbstractArray{T,N},
    xmean::Union{Nothing,AbstractArray{T,N}},
    xvar::Union{Nothing,AbstractArray{T,N}},
    scale::Union{Nothing,AbstractArray{T,N}},
    bias::Union{Nothing,AbstractArray{T,N}},
    activation::AT;
    training::Bool,
) where {T,N,affine,track_stats,AT}
    reduce_dims = get_reduce_dims(l, x)
    if !training
        # Computing the mean and variance for the batch
        if !track_stats
            batchmean = mean(x; dims=reduce_dims)
            batchvar = var(x; mean=batchmean, dims=reduce_dims, corrected=false)
        else
            batchmean = xmean
            batchvar = xvar
        end
    else
        batchmean = mean(x; dims=reduce_dims)
        batchvar = var(x; mean=batchmean, dims=reduce_dims, corrected=false)

        if track_stats
            xmean, xvar = update_statistics(
                l, xmean, xvar, batchmean, batchvar, l.momentum, T(prod(size(x, i) for i in reduce_dims))
            )
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

# Convolution
@inline conv_wrapper(x, weight, cdims) = conv(x, weight, cdims)

@inline function conv_wrapper(x::SubArray{T,N,<:CuArray}, weight, cdims) where {T,N}
    return conv(copy(x), weight, cdims)
end

# Dropout
function dropout(rng::AbstractRNG, x, prob, dims, training)
    if training
        rng = replicate(rng)
        mask = _dropout_mask(rng, x, prob; dims)
        return x .* mask, mask, rng
    else
        # Return `x` for type stability
        return x, x, rng
    end
end

# Adaptive Pooling
function compute_adaptive_pooling_dims(x::AbstractArray, outsize)
    insize = size(x)[1:end-2]
    stride = insize .÷ outsize
    k = insize .- (outsize .- 1) .* stride
    pad = 0
    return PoolDims(x, k; padding=pad, stride=stride)
end
