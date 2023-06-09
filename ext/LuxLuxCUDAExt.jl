module LuxLuxCUDAExt

isdefined(Base, :get_extension) ? (using LuxCUDA) : (using ..LuxCUDA)
using Lux, LuxLib, Random

function __init__()
    Lux.ACCELERATOR_STATE_CHANGED[] = true
    return
end

# utils.jl
Lux.replicate(rng::CUDA.RNG) = deepcopy(rng)

@inline function Lux._init_hidden_state(rng::AbstractRNG,
    rnn,
    x::Union{CUDA.StridedSubCuArray, CuArray})
    return CuArray(rnn.init_state(rng, rnn.out_dims, size(x, 2)))
end

@inline function Lux._conv(x::SubArray{T, N, <:CuArray}, weight, cdims) where {T, N}
    return conv(copy(x), weight, cdims)
end

@inline function Lux._conv_transpose(x::SubArray{T, N, <:CuArray},
    weight,
    cdims) where {T, N}
    return ∇conv_data(copy(x), weight, cdims)
end

# adapt.jl

end
