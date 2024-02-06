module LuxReverseDiffExt

using Lux, ReverseDiff

Lux.__value(x::AbstractArray{<:ReverseDiff.TrackedReal}) = ReverseDiff.value.(x)
Lux.__value(x::ReverseDiff.TrackedArray) = ReverseDiff.value(x)

end
