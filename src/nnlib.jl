## TODO: Eventually we want to move all these functions and their adjoints to NNlib.jl

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

## FIXME: Zygote doesn't like these branching. We can compile these away pretty easily
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
@inline _dropout_shape(s, ::Colon) = size(s)
@inline _dropout_shape(s, dims) = tuple((i ∉ dims ? 1 : si for (i, si) in enumerate(size(s)))...)

## TODO: Cache `1 / q` since we never need `q`
@inline _dropout_kernel(y::T, p, q) where {T} = y > p ? T(1 / q) : T(0)

@inline function generate_dropout_mask(rng::AbstractRNG, x, p; dims=:)
    realfptype = float(real(eltype(x)))
    y = rand!(rng, similar(x, realfptype, _dropout_shape(x, dims)))
    y .= _dropout_kernel.(y, p, 1 - p)
    return y
end

@inline function dropout(rng::AbstractRNG, x, prob, dims, training)
    if training
        rng = replicate(rng)
        mask = generate_dropout_mask(rng, x, prob; dims)
        return applydropout(x, mask), mask, rng
    else
        # Return `x` for type stability
        return x, x, rng
    end
end

@inline applydropout(x, mask) = x .* mask

# Adaptive Pooling
@inline function compute_adaptive_pooling_dims(x::AbstractArray, outsize)
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

getCUDNNActivationMode(::Union{typeof(tanh),typeof(tanh_fast)}) = CUDNN.CUDNN_ACTIVATION_TANH
getCUDNNActivationMode(::Union{typeof(sigmoid),typeof(sigmoid_fast)}) = CUDNN.CUDNN_ACTIVATION_SIGMOID
getCUDNNActivationMode(::Union{typeof(relu)}) = CUDNN.CUDNN_ACTIVATION_RELU
getCUDNNActivationMode(::Union{typeof(elu)}) = CUDNN.CUDNN_ACTIVATION_ELU

"""
    applyactivation(f::Function, x::AbstractArray)

Apply the function `f` on `x` elementwise, i.e. `f.(x)`. Dispatches to CUDNN if possible.
"""
@inline applyactivation(f::Function, x::AbstractArray) = f.(x)
@inline function applyactivation(f::cudnnValidActivationTypes, x::CuArray)
    return CUDNN.cudnnActivationForward(x; mode=getCUDNNActivationMode(f))
end
@inline applyactivation(::typeof(identity), x::AbstractArray) = x

# Dispatch Certain Broadcasted Functions to CUDNN
@inline function broadcast_shape_pullback(x, Δ)
    sx = size(x)
    sΔ = size(Δ)
    sx == sΔ && return Δ
    return sum(Δ; dims=findall(sx .!= sΔ))
end

@inline isvalidtensorop(x1, x2) = false
@inline function isvalidtensorop(x1::CuArray{N,T}, x2::CuArray{N,T}) where {N,T}
    return ndims(x1) <= 5 && (all(size(x2, i) == size(x1, i) || size(x2, i) == 1 for i in 1:ndims(x2)))
end

"""
    elementwise_add(x, y)

Computes `x .+ y`. Dispatches to CUDNN if possible
"""
@inline elementwise_add(x, y) = x .+ y
@inline function elementwise_add(x::CuArray, y::CuArray)
    !isvalidtensorop(x, y) && return x .+ y
    return CUDNN.cudnnOpTensor(x, y; op=CUDNN.CUDNN_OP_TENSOR_ADD)
end

@inline elementwise_add_pullback(x, y, Δ) = broadcast_shape_pullback(x, Δ), broadcast_shape_pullback(y, Δ)

"""
    elementwise_mul(x, y)

Computes `x .* y`. Dispatches to CUDNN if possible
"""
@inline elementwise_mul(x, y) = x .* y
@inline function elementwise_mul(x::CuArray, y::CuArray)
    !isvalidtensorop(x, y) && return x .* y
    return CUDNN.cudnnOpTensor(x, y; op=CUDNN.CUDNN_OP_TENSOR_MUL)
end

@inline function elementwise_mul_pullback(x, y, Δ)
    return broadcast_shape_pullback(x, elementwise_mul(Δ, y)), broadcast_shape_pullback(y, elementwise_mul(Δ, x))
end

# CUDNN Helpers
function cudnnOpTensorWithDefaults(
    x1,
    x2;
    y=similar(x1),
    op::CUDNN.cudnnOpTensorOp_t=CUDNN.CUDNN_OP_TENSOR_ADD,
    compType::DataType=(eltype(x1) <: Float64 ? Float64 : Float32),
    nanOpt::CUDNN.cudnnNanPropagation_t=CUDNN.CUDNN_NOT_PROPAGATE_NAN,
    opTensorDesc::CUDNN.cudnnOpTensorDescriptor=CUDNN.cudnnOpTensorDescriptor(
        op, CUDNN.cudnnDataType(compType), nanOpt
    ),
    alpha1::Real=1,
    alpha2::Real=1,
    beta::Real=0,
    x1Desc::CUDNN.cudnnTensorDescriptor=CUDNN.cudnnTensorDescriptor(x1),
    x2Desc::CUDNN.cudnnTensorDescriptor=CUDNN.cudnnTensorDescriptor(x2),
    yDesc::CUDNN.cudnnTensorDescriptor=CUDNN.cudnnTensorDescriptor(y),
)
    T = eltype(x1)
    alpha1, alpha2, beta = scalingParameter(T, alpha1), scalingParameter(T, alpha2), scalingParameter(T, beta)
    return CUDNN.cudnnOpTensorAD(x1, x2; opTensorDesc, alpha1, x1Desc, alpha2, x2Desc, beta, yDesc, y)
end

function cudnnActivationBackward(y::CuArray{T}, Δ::CuArray{T}, x::CuArray{T}; mode) where {T}
    Δx = similar(x)
    desc = CUDNN.cudnnActivationDescriptor(mode, CUDNN.CUDNN_NOT_PROPAGATE_NAN, Cdouble(1))
    CUDNN.cudnnActivationBackward(
        CUDNN.handle(),
        desc,
        CUDNN.scalingParameter(T, 1),
        CUDNN.cudnnTensorDescriptor(y),
        y,
        CUDNN.cudnnTensorDescriptor(Δ),
        Δ,
        CUDNN.cudnnTensorDescriptor(x),
        x,
        CUDNN.scalingParameter(T, 0),
        CUDNN.cudnnTensorDescriptor(Δx),
        Δx,
    )
    return Δx
end