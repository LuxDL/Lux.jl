module ReactantExt

using Reactant: Reactant, AnyTracedRArray, AnyTracedRVector, TracedRArray, @opcall
using ReactantCore: materialize_traced_array
using Static: False, True

using LuxLib: LuxLib, Impl, Optional, Utils

include("attention.jl")
include("batched_mul.jl")
include("batchnorm.jl")

end
