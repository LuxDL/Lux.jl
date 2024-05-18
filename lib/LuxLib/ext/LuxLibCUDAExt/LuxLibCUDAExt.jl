module LuxLibCUDAExt

# This file only wraps functionality part of CUDA like CUBLAS
using CUDA: CUDA, CUBLAS, StridedCuMatrix, StridedCuVector, CuPtr, AnyCuMatrix, AnyCuVector
using ChainRulesCore: ChainRulesCore
using FastClosures: @closure
using LinearAlgebra: LinearAlgebra, Transpose, Adjoint
using LuxLib: LuxLib
using NNlib: NNlib

const CRC = ChainRulesCore

const cuBLASLt_functional = Ref(true)

function __init__()
    try
        # Test if cuBLASLt is functional
        y = CUDA.zeros(Float32, 2, 2)
        w = CUDA.rand(Float32, 2, 2)
        x = CUDA.rand(Float32, 2, 2)
        b = CUDA.rand(Float32, 2)
        LuxLib._cublaslt_matmul_fused!(y, identity, w, x, b)
    catch
        cuBLASLt_functional[] = false
    end

    if CUDA.functional() && !cuBLASLt_functional[]
        @warn "cuBLASLt is not functional on this system. We won't be able to use \
               optimized implementations of certain matmul operations."
    end
end

# Low level functions
include("cublaslt.jl")

# fused dense
include("fused_dense.jl")

end
