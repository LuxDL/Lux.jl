module LuxDeviceUtilsLuxCUDAExt

isdefined(Base, :get_extension) ? (using LuxCUDA) : (using ..LuxCUDA)
using ChainRulesCore, LuxDeviceUtils, Random
import Adapt: adapt_storage, adapt
import ChainRulesCore as CRC

function __init__()
    LuxDeviceUtils.ACCELERATOR_STATE_CHANGED[] = true
    return
end

# Device Transfer
## To GPU
adapt_storage(::LuxCUDAAdaptor, x) = cu(x)
adapt_storage(::LuxCUDAAdaptor, rng::AbstractRNG) = rng

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
