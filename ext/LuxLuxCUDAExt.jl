module LuxLuxCUDAExt

import Lux
import LuxCUDA: CUDA

Lux.__is_extension_loaded(::Val{:LuxCUDA}) = Val(true)

Lux.__set_device!(::Val{:CUDA}, id::Int) = CUDA.functional() && CUDA.device!(id)
function Lux.__set_device!(::Val{:CUDA}, ::Nothing, rank::Int)
    CUDA.functional() || return
    CUDA.device!(rank % length(CUDA.devices()))
    return
end

end
