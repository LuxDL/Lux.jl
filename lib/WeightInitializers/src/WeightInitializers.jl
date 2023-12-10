module WeightInitializers

using PartialFunctions, Random, SpecialFunctions, Statistics

import PackageExtensionCompat: @require_extensions
function __init__()
    @require_extensions
end

include("utils.jl")
include("initializers.jl")

export zeros64, ones64, rand64, randn64, zeros32, ones32, rand32, randn32, zeros16, ones16,
    rand16, randn16
export zerosC64, onesC64, randC64, randnC64, zerosC32, onesC32, randC32, randnC32, zerosC16,
    onesC16, randC16, randnC16
export glorot_normal, glorot_uniform
export kaiming_normal, kaiming_uniform
export truncated_normal

end
