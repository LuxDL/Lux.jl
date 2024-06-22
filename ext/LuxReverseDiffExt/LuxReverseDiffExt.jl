module LuxReverseDiffExt

using ADTypes: ADTypes, AutoReverseDiff
using ArrayInterface: ArrayInterface
using Functors: fmap
using Lux: Lux, LuxCPUDevice
using ReverseDiff: ReverseDiff, TrackedArray, @grad_from_chainrules
using Setfield: @set!

# AoS to SoA conversion
function Lux.apply(
        m::Lux.AbstractExplicitLayer, x::AbstractArray{<:ReverseDiff.TrackedReal}, ps, st)
    @warn "Lux.apply(m::Lux.AbstractExplicitLayer, \
           x::AbstractArray{<:ReverseDiff.TrackedReal}, ps, st) input was corrected to \
           Lux.apply(m::Lux.AbstractExplicitLayer, x::ReverseDiff.TrackedArray}, ps, \
           st).\n\n\
           1. If this was not the desired behavior overload the dispatch on `m`.\n\n\
           2. This might have performance implications. Check which layer was causing this \
              problem using `Lux.Experimental.@debug_mode`." maxlog=1
    return Lux.apply(m, reshape(ArrayInterface.aos_to_soa(x), size(x)), ps, st)
end

## Prevent an infinite loop
Lux.apply(m::Lux.AbstractExplicitLayer, x::TrackedArray, ps, st) = m(x, ps, st)

include("rules.jl")
include("training.jl")

end
