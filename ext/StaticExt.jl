module StaticExt

using Lux: Lux, LuxOps, NilSizePropagation, Utils
using Lux.NilSizePropagation: Nil, NIL_STATIC_ERROR_MSG
using Static: StaticBool, StaticFloat64, StaticInt, StaticSymbol

Utils.static(::StaticBool{B}) where {B} = Val(B)
Utils.static(::StaticInt{N}) where {N} = Val(N)
Utils.static(::StaticSymbol{S}) where {S} = Val(S)

Utils.known(::StaticBool{B}) where {B} = B
Utils.known(::StaticInt{N}) where {N} = N
Utils.known(::StaticSymbol{S}) where {S} = S
Utils.known(::Type{<:StaticBool{B}}) where {B} = B
Utils.known(::Type{<:StaticInt{N}}) where {N} = N
Utils.known(::Type{<:StaticSymbol{S}}) where {S} = S

LuxOps.getproperty(x, ::StaticSymbol{v}) where {v} = LuxOps.getproperty(x, Val(v))
function LuxOps.getproperty(x::NamedTuple, ::StaticSymbol{v}) where {v}
    return LuxOps.getproperty(x, Val(v))
end

LuxOps.istraining(::StaticBool{B}) where {B} = B

Lux.recursive_eltype(x, ::StaticBool{B}) where {B} = Lux.recursive_eltype(x, Val(B))

Lux.Upsample(mode::StaticSymbol; kwargs...) = Lux.Upsample(Utils.static(mode); kwargs...)
Lux.Upsample(scale, mode::StaticSymbol) = Lux.Upsample(scale, Utils.static(mode))

function Lux.SimpleChainsLayer(layer, to_array::StaticBool)
    return Lux.SimpleChainsLayer(layer, Utils.static(to_array))
end

function Lux.ToSimpleChainsAdaptor(input_dims, convert_to_array::StaticBool)
    return Lux.ToSimpleChainsAdaptor(input_dims, Utils.static(convert_to_array))
end

function Base.convert(
    ::Type{Nil}, ::Union{StaticBool{N},StaticFloat64{N},StaticInt{N}}
) where {N}
    throw(ArgumentError(NIL_STATIC_ERROR_MSG))
end

end
