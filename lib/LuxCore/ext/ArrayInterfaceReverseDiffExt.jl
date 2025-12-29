module ArrayInterfaceReverseDiffExt

using ArrayInterface: ArrayInterface
using LuxCore: LuxCore, AbstractLuxLayer
using ReverseDiff: TrackedReal, TrackedArray

# AoS to SoA conversion
function LuxCore.apply(m::AbstractLuxLayer, x::AbstractArray{<:TrackedReal}, ps, st)
    @warn "Lux.apply(m::AbstractLuxLayer, \
           x::AbstractArray{<:ReverseDiff.TrackedReal}, ps, st) input was corrected to \
           Lux.apply(m::AbstractLuxLayer, x::ReverseDiff.TrackedArray}, ps, st).\n\n\
        1. If this was not the desired behavior overload the dispatch on `m`.\n\n\
        2. This might have performance implications. Check which layer was causing this \
           problem using `Lux.Experimental.@debug_mode`." maxlog = 1
    return LuxCore.apply(m, reshape(ArrayInterface.aos_to_soa(x), size(x)), ps, st)
end

## Prevent an infinite loop
LuxCore.apply(m::AbstractLuxLayer, x::TrackedArray, ps, st) = m(x, ps, st)

end
