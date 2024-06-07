module LuxDeviceUtilsLuxCUDAExt

using LuxCUDA: LuxCUDA
using LuxDeviceUtils: LuxDeviceUtils, LuxCUDADevice, reset_gpu_device!

__init__() = reset_gpu_device!()

LuxDeviceUtils.loaded(::Union{LuxCUDADevice, Type{<:LuxCUDADevice}}) = true
function LuxDeviceUtils.functional(::Union{LuxCUDADevice, Type{<:LuxCUDADevice}})
    return LuxCUDA.functional()
end

end
