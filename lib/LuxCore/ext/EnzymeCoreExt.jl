module EnzymeCoreExt

using EnzymeCore: EnzymeCore, EnzymeRules
using LuxCore: LuxCore
using Random: AbstractRNG

EnzymeRules.inactive(::typeof(LuxCore.replicate), ::AbstractRNG) = nothing

# Handle common mistakes users might make
const LAYER_DERIVATIVE_ERROR_MSG = """
Lux Layers only support `EnzymeCore.Const` annotation.

Lux Layers are immutable constants and gradients w.r.t. them are `nothing`. To
compute the gradients w.r.t. the layer's parameters, use the first argument returned
by `LuxCore.setup(rng, layer)` instead.
"""

function EnzymeCore.Active(::LuxCore.AbstractLuxLayer)
    throw(ArgumentError(LAYER_DERIVATIVE_ERROR_MSG))
end

for annotation in (:Duplicated, :DuplicatedNoNeed)
    @eval function EnzymeCore.$(annotation)(
        ::LuxCore.AbstractLuxLayer, ::LuxCore.AbstractLuxLayer
    )
        throw(ArgumentError(LAYER_DERIVATIVE_ERROR_MSG))
    end
end

for annotation in (:BatchDuplicated, :BatchDuplicatedNoNeed)
    @eval function EnzymeCore.$(annotation)(
        ::LuxCore.AbstractLuxLayer, ::NTuple{N,<:LuxCore.AbstractLuxLayer}, check::Bool=true
    ) where {N}
        throw(ArgumentError(LAYER_DERIVATIVE_ERROR_MSG))
    end
end

end
