module LuxLibForwardDiffExt

using ForwardDiff, LuxLib

function LuxLib._dropout_fptype(x::AbstractArray{<:ForwardDiff.Dual})
    return ForwardDiff.valtype(eltype(x))
end

end
