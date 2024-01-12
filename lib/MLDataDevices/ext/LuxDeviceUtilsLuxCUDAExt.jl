module LuxDeviceUtilsLuxCUDAExt

using ChainRulesCore, LuxCUDA, LuxDeviceUtils, Random
import Adapt: adapt_storage, adapt
import ChainRulesCore as CRC

__init__() = reset_gpu_device!()

LuxDeviceUtils.__is_loaded(::LuxCUDADevice) = true
LuxDeviceUtils.__is_functional(::LuxCUDADevice) = LuxCUDA.functional()

# Default RNG
device_default_rng(::LuxCUDADevice) = CUDA.default_rng()

# Device Transfer
## To GPU
adapt_storage(::LuxCUDAAdaptor, x) = cu(x)
adapt_storage(::LuxCUDAAdaptor, rng::AbstractRNG) = rng
adapt_storage(::LuxCUDAAdaptor, rng::Random.TaskLocalRNG) = CUDA.default_rng()

adapt_storage(::LuxCPUAdaptor, rng::CUDA.RNG) = Random.default_rng()

## To CPU
adapt_storage(::LuxCPUAdaptor, x::CUSPARSE.AbstractCuSparseMatrix) = adapt(Array, x)

## Chain Rules
CRC.rrule(::Type{Array}, x::CuArray) = Array(x), Δ -> (NoTangent(), cu(Δ))

function CRC.rrule(::typeof(adapt_storage), to::LuxCPUAdaptor, x::CUDA.AnyCuArray)
    function ∇adapt_storage(Δ)
        return (NoTangent(), NoTangent(), adapt_storage(LuxCUDAAdaptor(), Δ))
    end
    return adapt_storage(to, x), ∇adapt_storage
end

function CRC.rrule(::typeof(adapt_storage), to::LuxCUDAAdaptor, x::Array)
    function ∇adapt_storage(Δ)
        return (NoTangent(), NoTangent(), adapt_storage(LuxCPUAdaptor(), Δ))
    end
    return adapt_storage(to, x), ∇adapt_storage
end

end
