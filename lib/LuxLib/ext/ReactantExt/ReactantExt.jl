module ReactantExt

using Reactant: Reactant, AnyTracedRArray, AnyTracedRVector, TracedRArray, @opcall
using ReactantCore: materialize_traced_array
using EnzymeCore: EnzymeCore
using Static: False, True
using Statistics: mean, var

using LuxLib: LuxLib, Impl, Optional, Utils

include("batched_mul.jl")
include("batchnorm.jl")
include("normalization.jl")

end
