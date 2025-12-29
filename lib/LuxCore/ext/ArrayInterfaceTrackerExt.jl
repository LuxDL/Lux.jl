module ArrayInterfaceTrackerExt

using ArrayInterface: ArrayInterface
using LuxCore: LuxCore, AbstractLuxLayer
using Tracker: TrackedReal, TrackedArray

# AoS to SoA conversion
function LuxCore.apply(m::AbstractLuxLayer, x::AbstractArray{<:TrackedReal}, ps, st)
    @warn "LuxCore.apply(m::AbstractLuxLayer, \
           x::AbstractArray{<:Tracker.TrackedReal}, ps, st) input was corrected to \
           LuxCore.apply(m::AbstractLuxLayer, x::Tracker.TrackedArray}, ps, st).\n\n\
           1. If this was not the desired behavior overload the dispatch on `m`.\n\n\
           2. This might have performance implications. Check which layer was causing this \
              problem using `Lux.Experimental.@debug_mode`." maxlog = 1
    return LuxCore.apply(m, ArrayInterface.aos_to_soa(x), ps, st)
end

## Prevent an infinite loop
LuxCore.apply(m::AbstractLuxLayer, x::TrackedArray, ps, st) = m(x, ps, st)

end
