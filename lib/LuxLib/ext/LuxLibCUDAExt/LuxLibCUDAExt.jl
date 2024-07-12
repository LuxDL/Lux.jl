module LuxLibCUDAExt

# This file only wraps functionality part of CUDA like CUBLAS
using CUDA: CUDA, CUBLAS, StridedCuMatrix, StridedCuVector, CuPtr, AnyCuMatrix, AnyCuVector
using ChainRulesCore: ChainRulesCore
using DispatchDoctor: @stable
using FastClosures: @closure
using LinearAlgebra: LinearAlgebra, Transpose, Adjoint
using LuxLib: LuxLib, Optional
using NNlib: NNlib

const CRC = ChainRulesCore

# Low level functions
include("cublaslt.jl")

# fused dense
include("fused_dense.jl")

end
