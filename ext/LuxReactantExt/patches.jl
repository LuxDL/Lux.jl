# For some reason xlogx and xlogy with boolean inputs leads to incorrect results sometimes
# XXX: Once https://github.com/EnzymeAD/Reactant.jl/pull/278 is merged and tagged
LuxOps.xlogx(x::TracedRNumber{Bool}) = zero(x)

function LuxOps.xlogy(x::TracedRNumber, y::TracedRNumber)
    return invoke(LuxOps.xlogy, Tuple{Number, Number}, float(x), float(y))
end
