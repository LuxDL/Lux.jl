module LuxCUDAExt

using LuxCUDA, ..Lux
using Adapt, ChainRulesCore, Functors, Lux, Random
using ..Lux: GPU_BACKEND, LuxCPUAdaptor, LuxCUDAAdaptor
import Adapt: adapt, adapt_storage
import ChainRulesCore as CRC

__init__() = Lux.set_backend_if_higher_priority!(LuxCUDAAdaptor())

# device.jl
function _check_cuda_gpu()
    if !LuxCUDA.functional()
        @warn """
        `gpu` function is called for CUDA but LuxCUDA.jl is not functional.
        Defaulting back to the CPU. (No action is required if you want to run on the CPU).
        """ maxlog=1
    end
    return nothing
end

CRC.@non_differentiable _check_cuda_gpu()
CRC.@non_differentiable LuxCUDA.functional()

function Lux.gpu(::LuxCUDAAdaptor, x)
    _check_cuda_gpu()

    return LuxCUDA.functional() ?
           fmap(x -> adapt(LuxCUDAAdaptor(), x), x; exclude=Lux._isleaf) : x
end

adapt_storage(::LuxCUDAAdaptor, x) = cu(x)
adapt_storage(::LuxCUDAAdaptor, rng::AbstractRNG) = rng
adapt_storage(::LuxCPUAdaptor, x::CUDA.CUSPARSE.AbstractCuSparseMatrix) = adapt(Array, x)

# utils.jl

# chainrules.jl

end
