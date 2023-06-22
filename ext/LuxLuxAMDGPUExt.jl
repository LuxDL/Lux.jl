module LuxLuxAMDGPUExt

isdefined(Base, :get_extension) ? (using LuxAMDGPU) : (using ..LuxAMDGPU)
using ChainRulesCore, Lux, LuxLib, Random
import Adapt: adapt_storage, adapt
import ChainRulesCore as CRC

function __init__()
    Lux.ACCELERATOR_STATE_CHANGED[] = true
    return
end

# utils.jl
Lux.replicate(rng::AMDGPU.rocRAND.RNG) = deepcopy(rng)

@inline function Lux._init_hidden_state(rng::AbstractRNG, rnn, x::AMDGPU.AnyROCArray)
    return ROCArray(rnn.init_state(rng, rnn.out_dims, size(x, 2)))
end

@inline function Lux._conv(x::SubArray{T, N, <:AMDGPU.AnyROCArray},
    weight,
    cdims) where {T, N}
    return conv(copy(x), weight, cdims)
end

@inline function Lux._conv_transpose(x::SubArray{T, N, <:AMDGPU.AnyROCArray},
    weight,
    cdims) where {T, N}
    return ∇conv_data(copy(x), weight, cdims)
end

# Device Transfer
## To GPU
adapt_storage(::Lux.LuxAMDGPUAdaptor, x) = roc(x)
adapt_storage(::Lux.LuxAMDGPUAdaptor, rng::AbstractRNG) = rng

## Chain Rules
CRC.rrule(::Type{Array}, x::ROCArray) = Array(x), Δ -> (NoTangent(), roc(Δ))

function CRC.rrule(::typeof(adapt_storage), to::Lux.LuxCPUAdaptor, x::AMDGPU.AnyROCArray)
    function ∇adapt_storage(Δ)
        return (NoTangent(), NoTangent(), adapt_storage(Lux.LuxAMDGPUAdaptor(), Δ))
    end
    return adapt_storage(to, x), ∇adapt_storage
end

function CRC.rrule(::typeof(adapt_storage), to::Lux.LuxAMDGPUAdaptor, x::Array)
    function ∇adapt_storage(Δ)
        return (NoTangent(), NoTangent(), adapt_storage(Lux.LuxCPUAdaptor(), Δ))
    end
    return adapt_storage(to, x), ∇adapt_storage
end

end
