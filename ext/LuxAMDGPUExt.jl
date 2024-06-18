module LuxAMDGPUExt

using AMDGPU: AMDGPU
using Lux: Lux

# Flux modifies Conv weights while mapping to AMD GPU
function Lux._maybe_flip_conv_weight(x::AMDGPU.AnyROCArray)
    # This is a very rare operation, hence we dont mind allowing scalar operations
    return AMDGPU.@allowscalar reverse(x; dims=ntuple(identity, ndims(x) - 2))
end

end
