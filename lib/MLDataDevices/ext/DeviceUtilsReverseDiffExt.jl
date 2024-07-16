module DeviceUtilsReverseDiffExt

using DeviceUtils: DeviceUtils
using ReverseDiff: ReverseDiff

for op in (:_get_device, :_get_device_type)
    @eval begin
        function DeviceUtils.$op(x::ReverseDiff.TrackedArray)
            return DeviceUtils.$op(ReverseDiff.value(x))
        end
        function DeviceUtils.$op(x::AbstractArray{<:ReverseDiff.TrackedReal})
            return DeviceUtils.$op(ReverseDiff.value.(x))
        end
    end
end

end
