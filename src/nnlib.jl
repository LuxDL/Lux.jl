# Convolution
@inline _conv(x, weight, cdims) = conv(x, weight, cdims)

@inline function _conv(x::SubArray{T, N, <:CuArray}, weight, cdims) where {T, N}
    return conv(copy(x), weight, cdims)
end

@inline _conv_transpose(x, weight, cdims) = ∇conv_data(x, weight, cdims)

@inline function _conv_transpose(x::SubArray{T, N, <:CuArray}, weight, cdims) where {T, N}
    return ∇conv_data(copy(x), weight, cdims)
end

function _conv_transpose_dims(x::AbstractArray, weight::AbstractArray; padding, stride,
                              dilation, groups)
    # Calculate size of "input", from ∇conv_data()'s perspective...
    combined_pad = (padding[1:2:end] .+ padding[2:2:end])
    I = (size(x)[1:(end - 2)] .- 1) .* stride .+ 1 .+
        (size(weight)[1:(end - 2)] .- 1) .* dilation .- combined_pad
    C_in = size(weight)[end - 1] * groups
    batch_size = size(x)[end]
    # Create DenseConvDims() that looks like the corresponding conv()
    w_size = size(weight)
    return DenseConvDims((I..., C_in, batch_size), w_size; stride, padding, dilation,
                         groups)
end

# Adaptive Pooling
@inline function compute_adaptive_pooling_dims(x::AbstractArray, outsize)
    insize = size(x)[1:(end - 2)]
    stride = insize .÷ outsize
    k = insize .- (outsize .- 1) .* stride
    pad = 0
    return PoolDims(x, k; padding=pad, stride=stride)
end
