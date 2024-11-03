"""
    fused_dense_bias_activation(σ::F, weight::AbstractMatrix, x::AbstractMatrix,
        b::Optional{<:AbstractVector}) where {F}

Compute `σ.(weight * x .+ b)` with the best possible implementation available. Currently
this implementation attempts to minimize reallocations by reusing the output buffer for
multiple operations.

## Arguments

  - `σ`: Activation function
  - `weight`: Weight matrix
  - `x`: Input matrix
  - `b`: Bias vector (can be `nothing`)

## Notes on implementation

  - If any of the inputs, don't support setindexing (aka immutable arrays) we fallback to
    the generic non-mutating implementation.
  - Maximum memory reuse and operation fusion is guaranteed for ChainRules compatible AD
    backends or backends that support mutation. Backends like `Tracker` and `ReverseDiff`
    fallback to the generic implementation.
  - For CUDA Arrays, this uses a special fused implementation via cuBLASLt.
  - For small CPU Arrays, we use LoopVectorization.jl. On `x86_64` we use Octavian for
    medium sized matrices. This is overridden if special BLAS implementations are loaded
    (currently `MKL`, `AppleAccelerate`, and `BLISBLAS`).

!!! tip "Load `Octavian.jl`

    Loading `Octavian.jl` enables a polyalgorithm that uses different backends based on the
    input sizes.
"""
function fused_dense_bias_activation(σ::F, weight::AbstractMatrix, x::AbstractMatrix,
        b::Optional{<:AbstractVector}) where {F}
    return fused_dense_impl(select_fastest_activation(σ, weight, x, b), weight, x, b)
end
