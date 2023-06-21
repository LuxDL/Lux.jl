module WeightInitializers

using PartialFunctions, Random, SpecialFunctions, Statistics

include("utils.jl")
include("initializers.jl")

export zeros32, ones32, rand32, randn32
export glorot_normal, glorot_uniform
export kaiming_normal, kaiming_uniform
export truncated_normal

end
