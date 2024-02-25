module LuxComponentArraysReverseDiffExt

using ComponentArrays, ReverseDiff, Lux

const TCA{V, D, N, DA, A, Ax} = ReverseDiff.TrackedArray{
    V, D, N, ComponentArray{V, N, A, Ax}, DA}

Lux.__named_tuple(x::TCA) = NamedTuple(x)

end
