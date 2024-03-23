module LuxLuxAMDGPUExt

import LuxAMDGPU: AMDGPU
import Lux: _maybe_flip_conv_weight

# Flux modifies Conv weights while mapping to AMD GPU
function _maybe_flip_conv_weight(x::AMDGPU.AnyROCArray)
    # This is a very rare operation, hence we dont mind allowing scalar operations
    return AMDGPU.@allowscalar reverse(x; dims=ntuple(identity, ndims(x) - 2))
end

end
