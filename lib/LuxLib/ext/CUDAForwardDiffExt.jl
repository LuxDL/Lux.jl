module CUDAForwardDiffExt

using CUDA: CUDA, CuArray
using ForwardDiff: ForwardDiff
using LuxLib: LuxLib, Impl
using NNlib: NNlib

# CUDA-specific softmax/logsoftmax for Dual arrays
for op in (:logsoftmax, :softmax)
    dual_op = Symbol(op, :_dual)
    @eval function NNlib.$(op)(
        x::CuArray{<:ForwardDiff.Dual{Tag,T,P}}; dims=1
    ) where {Tag,T,P}
        return Impl.$(dual_op)(x; dims)
    end
end

end
