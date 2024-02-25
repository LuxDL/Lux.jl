module LuxLuxAMDGPUExt

import LuxAMDGPU: AMDGPU
import Lux: _maybe_flip_conv_weight
import LuxAMDGPU: AMDGPU

Lux.__is_extension_loaded(::Val{:LuxAMDGPU}) = Val(true)

Lux.__set_device!(::Val{:AMDGPU}, id::Int) = AMDGPU.functional() && AMDGPU.device!(id)
function Lux.__set_device!(::Val{:AMDGPU}, ::Nothing, rank::Int)
    AMDGPU.functional() || return
    AMDGPU.device!(rank % length(AMDGPU.devices()))
    return
end

# Flux modifies Conv weights while mapping to AMD GPU
function _maybe_flip_conv_weight(x::AMDGPU.AnyROCArray)
    # This is a very rare operation, hence we dont mind allowing scalar operations
    return AMDGPU.@allowscalar reverse(x; dims=ntuple(identity, ndims(x) - 2))
end

end
