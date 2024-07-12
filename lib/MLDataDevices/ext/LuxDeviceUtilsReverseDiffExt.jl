module LuxDeviceUtilsReverseDiffExt

using LuxDeviceUtils: LuxDeviceUtils, LuxCPUDevice
using ReverseDiff: ReverseDiff

LuxDeviceUtils.get_device(::ReverseDiff.TrackedArray) = LuxCPUDevice()
LuxDeviceUtils.get_device(::AbstractArray{<:ReverseDiff.TrackedReal}) = LuxCPUDevice()
LuxDeviceUtils._get_device_type(::ReverseDiff.TrackedArray) = LuxCPUDevice
LuxDeviceUtils._get_device_type(::AbstractArray{<:ReverseDiff.TrackedReal}) = LuxCPUDevice

end
