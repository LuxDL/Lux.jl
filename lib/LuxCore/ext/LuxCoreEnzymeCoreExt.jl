module LuxCoreEnzymeCoreExt

using EnzymeCore: EnzymeRules
using LuxCore: LuxCore
using Random: AbstractRNG

EnzymeRules.inactive(::typeof(LuxCore.replicate), ::AbstractRNG) = nothing

end
