module LuxAMDGPUExt

isdefined(Base, :get_extension) ? (using LuxAMDGPU) : (using ..LuxAMDGPU)
using Adapt, ChainRulesCore, Functors, Lux, Random
using Lux: GPU_BACKEND, LuxAMDGPUAdaptor
import Adapt: adapt, adapt_storage
import ChainRulesCore as CRC

__init__() = Lux.set_backend_if_higher_priority!(LuxAMDGPUAdaptor())

# device.jl
function _check_amd_gpu()
    if !LuxAMDGPU.functional()
        @warn """
        `gpu` function is called for AMDGPU but LuxAMDGPU.jl is not functional.
        Defaulting back to the CPU. (No action is required if you want to run on the CPU).
        """ maxlog=1
    end
    return nothing
end

CRC.@non_differentiable _check_amd_gpu()
CRC.@non_differentiable LuxAMDGPU.functional()

function Lux.gpu(::LuxAMDGPUAdaptor, x)
    _check_amd_gpu()

    return LuxAMDGPU.functional() ?
           fmap(x -> adapt(LuxAMDGPUAdaptor(), x), x; exclude=Lux._isleaf) : x
end

adapt_storage(::LuxAMDGPUAdaptor, x) = roc(x)
adapt_storage(::LuxAMDGPUAdaptor, rng::AbstractRNG) = rng

end