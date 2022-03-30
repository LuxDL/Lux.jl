using SparseArrays
using CUDA.CUSPARSE

import Base: getindex, size, show, +, -, *, /, zero, display, broadcast, broadcasted, materialize, getfield, convert, transpose, adjoint
import SparseArrays: AbstractSparseArray, AbstractSparseMatrixCSC, _checkbuffers, getcolptr, rowvals, nonzeros
import CUDA: cu
# We have a lot of flexibility for sparse matrices here since we can arbitrarily drop
# gradients and basic operations can be performed without having to check if both
# contrainers have values in the same place

"""
    EFLSparseMatrixCSC(mat::Union{<:AbstractSparseMatrixCSC, <:AbstractCuSparseMatrix})

A wrapper over sparse matrices supported only a small subset of operations. Also this
wrapper assumes that any operation performed preserves the existing sparsity pattern.
"""
struct EFLSparseMatrixCSC{S<:SparseArrays.AbstractSparseMatrix,Ti,Tv<:Integer} <: AbstractSparseMatrixCSC{Ti,Tv}
    mat::S

    function EFLSparseMatrixCSC(
        mat::Union{<:AbstractSparseMatrixCSC{Ti,Tv},<:AbstractCuSparseMatrix{Ti,Tv}}
    ) where {Ti,Tv<:Integer}
        return new{typeof(mat),Ti,Tv}(mat)
    end
end

function EFLSparseMatrixCSC(m::EFLSparseMatrixCSC{<:AbstractCuSparseMatrix}, nzvals::CuVector)
    return EFLSparseMatrixCSC(CuSparseMatrixCSC(copy(m.mat.colPtr), copy(m.mat.rowVal), nzvals, m.mat.dims))
end

function EFLSparseMatrixCSC(m::EFLSparseMatrixCSC{<:AbstractSparseMatrixCSC}, nzvals::AbstractVector)
    return EFLSparseMatrixCSC(SparseMatrixCSC(m.mat.m, m.mat.n, copy(m.mat.colptr), copy(m.mat.rowval), nzvals))
end

transpose(m::EFLSparseMatrixCSC) = EFLSparseMatrixCSC(permutedims(m.mat, (2, 1)))

adjoint(m::EFLSparseMatrixCSC) = EFLSparseMatrixCSC(permutedims(m.mat, (2, 1)))

convert(::Type{T}, m::EFLSparseMatrixCSC) where {T<:CUDA.AbstractGPUArray} = convert(T, m.mat)

for f in (:size, :_checkbuffers, :getcolptr, :rowvals, :nonzeros)
    @eval begin
        function ($f)(m::EFLSparseMatrixCSC, args...; kwargs...)
            return ($f)(m.mat, args...; kwargs...)
        end
    end
end

getindex(m::EFLSparseMatrixCSC, i::Integer, j::Integer) = getindex(m.mat, i, j)

function show(io::IO, ::MIME"text/plain", m::EFLSparseMatrixCSC)
    xnnz = length(nonzeros(m))
    print(io, length(m), "-element ", typeof(m), " with ", xnnz, " stored ", xnnz == 1 ? "entry" : "entries")
    if xnnz != 0
        println(io, ":")
        Base.print_array(IOContext(io, :typeinfo => eltype(m)), SparseMatrixCSC(m.mat))
    end
end

show(io::IO, m::EFLSparseMatrixCSC) = show(io, m.mat)

cu(m::EFLSparseMatrixCSC) = EFLSparseMatrixCSC(cu(m.mat))
Flux.cpu(m::EFLSparseMatrixCSC) = EFLSparseMatrixCSC(Flux.cpu(m.mat))

zero(m::EFLSparseMatrixCSC) = m .* zero(eltype(m))

*(::EFLSparseMatrixCSC, ::EFLSparseMatrixCSC) = error("matmul doesn't preserve sparsity pattern")
*(m::EFLSparseMatrixCSC{S,T}, n::Union{Matrix{T},CuMatrix{T}}) where {S,T} = m.mat * n
*(m::Union{Matrix{T},CuMatrix{T}}, n::EFLSparseMatrixCSC{S,T}) where {S,T} = m * n.mat

for op in (:+, :-, :*, :/)
    @eval begin
        function broadcasted(
            ::Base.Broadcast.BroadcastStyle, $(op)::typeof($(op)), m1::EFLSparseMatrixCSC, m2::EFLSparseMatrixCSC
        )
            return EFLSparseMatrixCSC(m1, materialize(broadcasted($(op), nonzeros(m1), nonzeros(m2))))
        end

        function broadcasted(
            ::Base.Broadcast.BroadcastStyle,
            $(op)::typeof($(op)),
            m::EFLSparseMatrixCSC,
            n::Union{Number,Base.Broadcast.Broadcasted},
        )
            return EFLSparseMatrixCSC(m, materialize(broadcasted($(op), nonzeros(m), n)))
        end

        function broadcasted(
            ::Base.Broadcast.BroadcastStyle,
            $(op)::typeof($(op)),
            n::Union{Number,Base.Broadcast.Broadcasted},
            m::EFLSparseMatrixCSC,
        )
            return EFLSparseMatrixCSC(m, materialize(broadcasted($(op), n, nonzeros(m))))
        end
    end
end

function broadcasted(::Base.Broadcast.BroadcastStyle, f, mat::EFLSparseMatrixCSC)
    return EFLSparseMatrixCSC(mat, materialize(broadcasted(f, nonzeros(mat))))
end

function Base.broadcasted(
    ::Base.Broadcast.BroadcastStyle, f, mat1::EFLSparseMatrixCSC, mat2::T
) where {T<:AbstractArray}
    return Base.materialize(broadcasted(f, convert(T, mat1), mat2))
end

function Base.broadcasted(
    ::Base.Broadcast.BroadcastStyle, f, mat1::T, mat2::EFLSparseMatrixCSC
) where {T<:AbstractArray}
    return Base.materialize(broadcasted(f, mat1, convert(T, mat2)))
end

function broadcasted(
    ::Base.Broadcast.BroadcastStyle, f::typeof(*), mat1::EFLSparseMatrixCSC, mat2::T
) where {T<:AbstractArray}
    nzvals =
        EFLSparseMatrixCSC(sparse(materialize(broadcasted(f, convert(T, mat1), mat2 .+ eps(eltype(mat2)))))) .-
        mat1 .* eps(eltype(mat2))
    return EFLSparseMatrixCSC(mat1, nonzeros(nzvals))
end

function broadcasted(
    ::Base.Broadcast.BroadcastStyle, f::typeof(*), mat1::T, mat2::EFLSparseMatrixCSC
) where {T<:AbstractArray}
    nzvals =
        EFLSparseMatrixCSC(sparse(materialize(broadcasted(f, mat1 .+ eps(eltype(mat1)), convert(T, mat2))))) .-
        mat2 .* eps(eltype(mat1))
    return EFLSparseMatrixCSC(mat2, nonzeros(nzvals))
end
