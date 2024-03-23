module LuxComponentArraysReverseDiffExt

using ComponentArrays: ComponentArray
using Lux: Lux
using ReverseDiff: TrackedArray

const TCA{V, D, N, DA, A, Ax} = TrackedArray{V, D, N, ComponentArray{V, N, A, Ax}, DA}

Lux.__named_tuple(x::TCA) = NamedTuple(x)

end
