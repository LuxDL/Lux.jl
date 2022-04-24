# Normalization Implementation
@inline function update_statistics(::AbstractNormalizationLayer, xmean, xvar, batchmean, batchvar, momentum, m)
    batchmean = mean(batchmean; dims=ndims(batchmean))
    batchvar = mean(batchvar; dims=ndims(batchvar))
    _xmean = @. (1 - momentum) * xmean + momentum * batchmean
    _xvar = @. (1 - momentum) * xvar + momentum * batchvar * (m / (m - 1))
    return (_xmean, _xvar)
end

@inline function update_statistics(::BatchNorm, xmean, xvar, batchmean, batchvar, momentum, m)
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
function generate_dropout_mask(rng::AbstractRNG, x, p; dims=:)
    realfptype = float(real(eltype(x)))
    y = rand!(rng, similar(x, realfptype, _dropout_shape(x, dims)))
    y .= _dropout_kernel.(y, p, 1 - p)
    return y
end

function dropout(rng::AbstractRNG, x, prob, dims, training)
    if training
        rng = replicate(rng)
        mask = generate_dropout_mask(rng, x, prob; dims)
        return applydropout(x, mask), mask, rng
    else
        # Return `x` for type stability
        return x, x, rng
    end
end

applydropout(x, mask) = x .* mask

# Adaptive Pooling
function compute_adaptive_pooling_dims(x::AbstractArray, outsize)
    insize = size(x)[1:(end - 2)]
    stride = insize .÷ outsize
    k = insize .- (outsize .- 1) .* stride
    pad = 0
    return PoolDims(x, k; padding=pad, stride=stride)
end

# Activation Functions
## I think this is handled by NNlibCUDA. But currently leaving here for
## benchmarking larger models
const cudnnValidActivationTypes = Union{
    typeof(tanh),typeof(sigmoid),typeof(relu),typeof(elu),typeof(tanh_fast),typeof(sigmoid_fast)
}

getCUDNNActivationMode(::Union{typeof(tanh),typeof(tanh_fast)}) = CUDA.CUDNN.CUDNN_ACTIVATION_TANH
getCUDNNActivationMode(::Union{typeof(sigmoid),typeof(sigmoid_fast)}) = CUDA.CUDNN.CUDNN_ACTIVATION_SIGMOID
getCUDNNActivationMode(::Union{typeof(relu)}) = CUDA.CUDNN.CUDNN_ACTIVATION_RELU
getCUDNNActivationMode(::Union{typeof(elu)}) = CUDA.CUDNN.CUDNN_ACTIVATION_ELU

@inline function applyactivation(f::Function, x, ::Val{true})
    x .= f.(x)
end
@inline applyactivation(f::Function, x, ::Val{false}) = f.(x)
@inline function applyactivation(f::cudnnValidActivationTypes, x, ::Val{true})
    return CUDA.CUDNN.cudnnActivationForward!(x, x; mode=getCUDNNActivationMode(f))
end
@inline function applyactivation(f::cudnnValidActivationTypes, x, ::Val{false})
    return CUDA.CUDNN.cudnnActivationForward(x; mode=getCUDNNActivationMode(f))
end
@inline applyactivation(::typeof(identity), x, ::Val{true}) = x
@inline applyactivation(::typeof(identity), x, ::Val{false}) = x
