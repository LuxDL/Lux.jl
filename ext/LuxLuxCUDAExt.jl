module LuxLuxCUDAExt

using ChainRulesCore, Lux, LuxCUDA, LuxLib, Random
import ChainRulesCore as CRC

# utils.jl
Lux.replicate(rng::CUDA.RNG) = deepcopy(rng)

@inline function Lux._init_hidden_state(rng::AbstractRNG, rnn, x::CUDA.AnyCuArray)
    return CuArray(rnn.init_state(rng, rnn.out_dims, size(x, 2)))
end

@inline function Lux._conv(x::SubArray{T, N, <:CUDA.AnyCuArray}, weight, cdims) where {T, N}
    return conv(copy(x), weight, cdims)
end

@inline function Lux._conv_transpose(x::SubArray{T, N, <:CUDA.AnyCuArray}, weight,
    cdims) where {T, N}
    return ∇conv_data(copy(x), weight, cdims)
end

end
