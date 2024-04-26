module LuxLibCUDAExt

# This file only wraps functionality part of CUDA like CUBLAS
using CUDA: CUDA, CUBLAS, StridedCuMatrix, StridedCuVector, CuPtr
using LinearAlgebra: LinearAlgebra, Transpose, Adjoint, mul!
using LuxLib: LuxLib
using NNlib: NNlib

# Low level functions
include("cublaslt.jl")

# fused dense
include("fused_dense.jl")

end
