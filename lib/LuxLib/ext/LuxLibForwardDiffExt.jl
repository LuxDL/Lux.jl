module LuxLibForwardDiffExt

isdefined(Base, :get_extension) ? (using ForwardDiff) : (using ..ForwardDiff)
using LuxLib

function LuxLib._dropout_fptype(x::AbstractArray{<:ForwardDiff.Dual})
    return ForwardDiff.valtype(eltype(x))
end

end
