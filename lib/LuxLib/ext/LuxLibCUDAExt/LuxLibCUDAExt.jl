module LuxLibCUDAExt

# This file only wraps functionality part of CUDA like CUBLAS
using CUDA: CUDA, CUBLAS, StridedCuMatrix, StridedCuVector, CuPtr, AnyCuMatrix, AnyCuVector
using LinearAlgebra: LinearAlgebra, Transpose, Adjoint
using LuxLib: LuxLib, Optional
using LuxLib.Utils: ofeltype_array
using NNlib: NNlib
using Static: True, False

# Low level functions
include("cublaslt.jl")

end
