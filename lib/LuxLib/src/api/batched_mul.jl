"""
    batched_matmul(x, y)

Computes the batched matrix multiplication of `x` and `y`.  For more details see the NNlib
documentation on `NNlib.batched_mul`. This function is mostly a wrapper around `batched_mul`
but attempts to be faster on CPUs.
"""
function batched_matmul(x::AbstractMatrix, y::AbstractArray{<:Any, 3})
    return batched_matmul(reshape(x, size(x)..., 1), y)
end

function batched_matmul(x::AbstractArray{<:Any, 3}, y::AbstractMatrix)
    return batched_matmul(x, reshape(y, size(y)..., 1))
end

function batched_matmul(x::AbstractArray{<:Any, 3}, y::AbstractArray{<:Any, 3})
    return __batched_matmul_impl(
        attempt_fast_implementation((x, y)), get_device_type((x, y)), x, y)
end
