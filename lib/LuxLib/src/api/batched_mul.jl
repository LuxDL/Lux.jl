"""
    batched_matmul(x::AbstractMatrix, y::AbstractArray{yT,3}) where {yT}
    batched_matmul(x::AbstractArray{xT,3}, y::AbstractMatrix) where {xT}
    batched_matmul(
        x::AbstractArray{xT,N},
        y::AbstractArray{yT,N};
        lhs_contracting_dim::Int=2,
        rhs_contracting_dim::Int=1,
        lhs_batching_dims::Dims{M}=ntuple(Base.Fix2(+, 2), Val(N - 2)),
        rhs_batching_dims::Dims{M}=ntuple(Base.Fix2(+, 2), Val(N - 2)),
    ) where {xT,yT,N,M}

Computes the batched matrix multiplication of `x` and `y`. The following types are supported
for `x` and `y`:

  - `AbstractMatrix` and `AbstractArray{<:Number,3}`: `x` is treated as a tensor with batch
    size `1`.
  - `AbstractArray{<:Number,3}` and `AbstractMatrix`: `y` is treated as a tensor with batch
    size `1`.
  - `AbstractArray{<:Number,N}` and `AbstractArray{<:Number,N}`: This is the general form.
    This case supports the keyword arguments. Size of each corresponding batch dimension
    must be equal (or `1`).

## Keyword Arguments

  - `lhs_contracting_dim`: The dimension along which `x` is contracted. Defaults to `2`.
  - `rhs_contracting_dim`: The dimension along which `y` is contracted. Defaults to `1`.
  - `lhs_batching_dims`: The batching dimensions of `x`. Defaults to the last N - 2
    dimensions.
  - `rhs_batching_dims`: The batching dimensions of `y`. Defaults to the last N - 2
    dimensions.

## Output Shape

The output shape of result `z` is computed as follows:

  - `size(z, 1)`: size of the non-contracting dimension of `x`.
  - `size(z, 2)`: size of the non-contracting dimension of `y`.
  - `size(z)[3:end]`: size of each of the resolved batching dimensions (i.e.
    `max(size(x, lhs_batching_dims[j]), size(y, rhs_batching_dims[j])))`) of `x` and `y`.

## Performance Considerations

  - **CPU**: `x` and `y` are converted to 3D tensors and then we attempt to use a custom
    implementation based on LoopVectorization. If this fails, we fall back to `NNlib.batched_mul`.

!!! tip "Load `LoopVectorization.jl` to get faster batched matrix multiplication"

    On CPUs loading LoopVectorization adds faster implementations of batched matrix
    multiplication.

  - **GPU**: `x` and `y` are converted to 3D tensors and then we call `NNlib.batched_mul`.
  - **Reactant**: We directly lower this function to `stablehlo.dot_general`.
"""
function batched_matmul(x::AbstractMatrix, y::AbstractArray{yT,3}) where {yT}
    return batched_matmul(expand_batchdim(x), y)
end

function batched_matmul(x::AbstractArray{xT,3}, y::AbstractMatrix) where {xT}
    return batched_matmul(x, expand_batchdim(y))
end

function batched_matmul(
    x::AbstractArray{xT,N}, y::AbstractArray{yT,N}; kwargs...
) where {xT,yT,N}
    return batched_matmul_impl(x, y; kwargs...)
end
