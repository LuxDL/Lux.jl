module CUDAExt

using CUDA: CUDA, cuBLAS, CuMatrix, CuVector, CuPtr
using LinearAlgebra: LinearAlgebra, Transpose, Adjoint
using LuxLib: LuxLib, Optional
using LuxLib.Utils: ofeltype_array
using NNlib: NNlib
using Static: True, False

# Keep the established internal spelling while importing CUDA.jl's non-deprecated binding.
const CUBLAS = cuBLAS

# Low level functions
include("cublaslt.jl")

end
