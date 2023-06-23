module LuxLuxAMDGPUExt

isdefined(Base, :get_extension) ? (using LuxAMDGPU) : (using ..LuxAMDGPU)
using ChainRulesCore, Lux, LuxLib, Random
import ChainRulesCore as CRC

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

@inline function Lux._eachslice(x::AMDGPU.AnyROCArray, ::Val{dims}) where {dims}
    # FIXME: This is not efficient but AMDGPU doesn't deal with views well
    return [copy(selectdim(x, dims, i)) for i in axes(x, dims)]
end

# Flux modifies Conv weights while mapping to AMD GPU
function Lux._maybe_flip_conv_weight(x::AMDGPU.AnyROCArray)
    # This is a very rare operation, hence we dont mind allowing scalar operations
    return AMDGPU.@allowscalar reverse(x; dims=ntuple(identity, ndims(x) - 2))
end

end
