module LuxLibCUDAExt

using CUDA: CUDA, CUBLAS, CuArray, StridedCuMatrix, StridedCuVector, CuPtr
using ForwardDiff: ForwardDiff
using LinearAlgebra: LinearAlgebra, Transpose, Adjoint
using LuxLib: LuxLib, Impl, Optional
using LuxLib.Utils: ofeltype_array
using NNlib: NNlib
using Static: True, False

# Hacky Type Piracy for ForwardDiff rules
for op in (:logsoftmax, :softmax)
    dual_op = Symbol(op, :_dual)
    @eval function NNlib.$(op)(
            x::CuArray{<:ForwardDiff.Dual{Tag, T, P}}; dims=1) where {Tag, T, P}
        return Impl.$(dual_op)(x; dims)
    end
end

# Low level functions
include("cublaslt.jl")

end
