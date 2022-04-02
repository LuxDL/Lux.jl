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

fast_matmul!(C::AbstractVecOrMat, A::AbstractMatrix, B::AbstractVecOrMat) = matmul!(C, A, B)

function fast_matmul!(
    C::CuVecOrMat,
    A::Union{<:CuMatrix{T1},<:Adjoint{T1,<:CuVecOrMat{T1}},<:Transpose{T1,<:CuVecOrMat{T1}}},
    B::Union{<:CuVecOrMat{T2},<:Adjoint{T2,<:CuVecOrMat{T2}},<:Transpose{T2,<:CuVecOrMat{T2}}},
) where {T1,T2}
    return mul!(C, A, B)
end

