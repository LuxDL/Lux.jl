module LuxDeviceUtilsLuxAMDGPUExt

using ChainRulesCore, LuxAMDGPU, LuxDeviceUtils, Random
import Adapt: adapt_storage, adapt
import ChainRulesCore as CRC

__init__() = reset_gpu_device!()

LuxDeviceUtils.__is_loaded(::LuxAMDGPUDevice) = true
LuxDeviceUtils.__is_functional(::LuxAMDGPUDevice) = LuxAMDGPU.functional()

# Default RNG
device_default_rng(::LuxAMDGPUDevice) = AMDGPU.rocrand_rng()

# Device Transfer
## To GPU
adapt_storage(::LuxAMDGPUAdaptor, x) = roc(x)
adapt_storage(::LuxAMDGPUAdaptor, rng::AbstractRNG) = rng
adapt_storage(::LuxAMDGPUAdaptor, rng::Random.TaskLocalRNG) = AMDGPU.rocrand_rng()

adapt_storage(::LuxCPUAdaptor, rng::AMDGPU.rocRAND.RNG) = Random.default_rng()

## Chain Rules
CRC.rrule(::Type{Array}, x::ROCArray) = Array(x), Δ -> (NoTangent(), roc(Δ))

function CRC.rrule(::typeof(adapt_storage), to::LuxCPUAdaptor, x::AMDGPU.AnyROCArray)
    function ∇adapt_storage(Δ)
        return (NoTangent(), NoTangent(), adapt_storage(LuxAMDGPUAdaptor(), Δ))
    end
    return adapt_storage(to, x), ∇adapt_storage
end

function CRC.rrule(::typeof(adapt_storage), to::LuxAMDGPUAdaptor, x::Array)
    function ∇adapt_storage(Δ)
        return (NoTangent(), NoTangent(), adapt_storage(LuxCPUAdaptor(), Δ))
    end
    return adapt_storage(to, x), ∇adapt_storage
end

end
