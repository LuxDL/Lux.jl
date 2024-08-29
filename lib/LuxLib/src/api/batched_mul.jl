"""
    batched_matmul(x, y)

Computes the batched matrix multiplication of `x` and `y`.  For more details see the NNlib
documentation on `NNlib.batched_mul`. This function is mostly a wrapper around `batched_mul`
but attempts to be faster on CPUs.
"""
function batched_matmul(x::AbstractMatrix, y::AbstractArray{yT, 3}) where {yT}
    return batched_matmul(expand_batchdim(x), y)
end

function batched_matmul(x::AbstractArray{xT, 3}, y::AbstractMatrix) where {xT}
    return batched_matmul(x, expand_batchdim(y))
end

function batched_matmul(x::AbstractArray{xT, 3}, y::AbstractArray{yT, 3}) where {xT, yT}
    return batched_matmul_impl(x, y)
end
