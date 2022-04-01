zeros32(rng::AbstractRNG, args...; kwargs...) = zeros32(args...; kwargs...)
ones32(rng::AbstractRNG, args...; kwargs...) = ones32(args...; kwargs...)
Base.zeros(rng::AbstractRNG, args...; kwargs...) = zeros(args...; kwargs...)
Base.ones(rng::AbstractRNG, args...; kwargs...) = ones(args...; kwargs...)

"""
    fast_matmul(A, B)

Dispatch to Octavian for CPU and CUBLAS for GPU
"""
fast_matmul(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T} =
    fast_matmul!(similar(A, (size(A, 1), size(B, 2))), A, B)

fast_matmul(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T} =
    fast_matmul!(similar(A, size(A, 1)), A, b)

fast_matmul!(C::AbstractVecOrMat{T}, A::AbstractMatrix{T}, B::AbstractVecOrMat{T}) where {T} =
    matmul!(C, A, B)

fast_matmul!(C::CuVecOrMat{T}, A::CuMatrix{T}, B::CuVecOrMat{T}) where {T} =
    mul!(C, A, B)
