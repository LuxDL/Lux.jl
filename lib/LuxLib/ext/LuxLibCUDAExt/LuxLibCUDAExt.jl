module LuxLibCUDAExt

# This file only wraps functionality part of CUDA like CUBLAS
using CUDA: CUDA, CUBLAS, StridedCuMatrix, StridedCuVector, CuPtr
using LinearAlgebra: LinearAlgebra, Transpose, Adjoint, mul!
using LuxLib: LuxLib
using NNlib: NNlib

# Low level functions
include("cublaslt.jl")

# fused dense
@inline __length(x) = length(x)
@inline __length(::Nothing) = nothing

function LuxLib.__fused_dense_bias_activation_impl(
        act::F, weight::CUDA.AnyCuMatrix, x::CUDA.AnyCuMatrix,
        b::Union{Nothing, CUDA.AnyCuVector}) where {F}
    y = similar(x, LuxLib.__get_concrete_fba_output_eltype(act, weight, x, b),
        size(weight, 1), size(x, 2))
    if hasmethod(LuxLib._cublaslt_matmul_fused!,
        (typeof(y), F, typeof(weight), typeof(x), typeof(b)))
        retcode = LuxLib._cublaslt_matmul_fused!(y, act, weight, x, b)
        retcode == 0 && return y
        # cuBLASLt failed for the given inputs use the generic fallback
        @warn "cuBLASLt failed for the given inputs $(act), $(typeof(weight)) \
               [$(size(weight))], $(typeof(x)) [$(size(x))], $(typeof(b)) \
               [$(__length(b))]. Falling back to generic implementation." maxlog=1
    end
    mul!(y, weight, x)
    return LuxLib.__apply_bias_activation!!(act, y, b, Val(false))
end

end
