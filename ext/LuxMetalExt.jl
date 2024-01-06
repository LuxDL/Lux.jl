module LuxMetalExt

using Lux, LuxLib, Metal, Random

@inline function Lux._init_hidden_state(rng::AbstractRNG, rnn, x::MtlArray)
    return MtlArray(rnn.init_state(rng, rnn.out_dims, size(x, 2)))
end

@inline function Lux._conv(x::SubArray{T, N, <:MtlArray}, weight, cdims) where {T, N}
    return conv(copy(x), weight, cdims)
end

@inline function Lux._conv_transpose(x::SubArray{T, N, <:MtlArray}, weight,
        cdims) where {T, N}
    return ∇conv_data(copy(x), weight, cdims)
end

@inline function Lux._eachslice(x::MtlArray, ::Val{dims}) where {dims}
    # FIXME: This is not efficient but Metal doesn't deal with views well
    return [copy(selectdim(x, dims, i)) for i in axes(x, dims)]
end

end
