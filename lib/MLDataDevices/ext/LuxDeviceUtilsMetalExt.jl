module LuxDeviceUtilsMetalExt

using ChainRulesCore, LuxDeviceUtils, Metal, Random
import Adapt: adapt_storage, adapt
import ChainRulesCore as CRC

__init__() = reset_gpu_device!()

LuxDeviceUtils.__is_loaded(::LuxMetalDevice) = true
LuxDeviceUtils.__is_functional(::LuxMetalDevice) = Metal.functional()

__default_rng() = Metal.GPUArrays.default_rng(MtlArray)

# Device Transfer
## To GPU
adapt_storage(::LuxMetalAdaptor, x) = mtl(x)
adapt_storage(::LuxMetalAdaptor, rng::AbstractRNG) = rng

@static if VERSION ≥ v"1.9-"
    adapt_storage(::LuxMetalAdaptor, rng::Random.TaskLocalRNG) = __default_rng()
else
    adapt_storage(::LuxMetalAdaptor, rng::Random.MersenneTwister) = __default_rng()
end

## Is this a correct thing to do?
adapt_storage(::LuxCPUAdaptor, rng::Metal.GPUArrays.RNG) = Random.default_rng()

## Chain Rules
CRC.rrule(::Type{Array}, x::MtlArray) = Array(x), Δ -> (NoTangent(), MtlArray(Δ))

function CRC.rrule(::typeof(adapt_storage), to::LuxCPUAdaptor, x::MtlArray)
    function ∇adapt_storage(Δ)
        return (NoTangent(), NoTangent(), adapt_storage(LuxMetalAdaptor(), Δ))
    end
    return adapt_storage(to, x), ∇adapt_storage
end

function CRC.rrule(::typeof(adapt_storage), to::LuxMetalAdaptor, x::Array)
    function ∇adapt_storage(Δ)
        return (NoTangent(), NoTangent(), adapt_storage(LuxCPUAdaptor(), Δ))
    end
    return adapt_storage(to, x), ∇adapt_storage
end

end
