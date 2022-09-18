module LuxLib

using ChainRulesCore, CUDA, CUDAKernels, KernelAbstractions, Markdown, NNlib, NNlibCUDA,
      Random, Statistics
import ChainRulesCore as CRC

include("utils.jl")

# Low-Level Implementations
include("impl/groupnorm.jl")
include("impl/normalization.jl")

# User Facing
include("api/batchnorm.jl")
include("api/dropout.jl")
include("api/groupnorm.jl")
include("api/layernorm.jl")

export batchnorm, groupnorm, layernorm
export dropout

end
