module LuxDeviceUtilsReverseDiffExt

using LuxDeviceUtils: LuxDeviceUtils
using ReverseDiff: ReverseDiff

@inline function LuxDeviceUtils.get_device(x::ReverseDiff.TrackedArray)
    return LuxDeviceUtils.get_device(ReverseDiff.value(x))
end
@inline function LuxDeviceUtils.get_device(x::AbstractArray{<:ReverseDiff.TrackedReal})
    return LuxDeviceUtils.get_device(ReverseDiff.value.(x))
end

end
