module LuxDeviceUtilsReverseDiffExt

using LuxDeviceUtils: LuxDeviceUtils
using ReverseDiff: ReverseDiff

for op in (:_get_device, :_get_device_type)
    @eval begin
        function LuxDeviceUtils.$op(x::ReverseDiff.TrackedArray)
            return LuxDeviceUtils.$op(ReverseDiff.value(x))
        end
        function LuxDeviceUtils.$op(x::AbstractArray{<:ReverseDiff.TrackedReal})
            return LuxDeviceUtils.$op(ReverseDiff.value.(x))
        end
    end
end

end
