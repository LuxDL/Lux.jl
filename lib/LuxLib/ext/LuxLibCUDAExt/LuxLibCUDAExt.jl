module LuxLibCUDAExt

# This file only wraps functionality part of CUDA like CUBLAS
using CUDA: CUDA, CUBLAS, StridedCuMatrix, StridedCuVector, CuPtr, AnyCuMatrix, AnyCuVector
using LinearAlgebra: LinearAlgebra, Transpose, Adjoint
using LuxLib: LuxLib, Optional
using NNlib: NNlib
using Static: True, False, known

# Low level functions
include("cublaslt.jl")

end
