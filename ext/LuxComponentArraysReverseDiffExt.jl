module LuxComponentArraysReverseDiffExt

if isdefined(Base, :get_extension)
    using ComponentArrays
    using ReverseDiff
else
    using ..ComponentArrays
    using ..ReverseDiff
end

using Lux

const TCA{V, D, N, DA, A, Ax} = ReverseDiff.TrackedArray{V, D, N,
                                                         ComponentArray{V, N, A, Ax}, DA}

@inline function Lux._getproperty(x::TCA, ::Val{prop}) where {prop}
    return prop in propertynames(ReverseDiff.value(x)) ? getproperty(x, prop) : nothing
end

# Freezing
function _get_named_tuple(x::TCA)
    fields = propertynames(ReverseDiff.value(x))
    return NamedTuple{fields}(getproperty.((x,), fields))
end

Lux._merge(nt1::TCA, nt2::NamedTuple) = merge(_get_named_tuple(nt1), nt2)
Lux._merge(nt1::NamedTuple, nt2::TCA) = merge(nt1, _get_named_tuple(nt2))

end
