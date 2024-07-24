module DeviceUtilsReverseDiffExt

using MLDataDevices: MLDataDevices
using ReverseDiff: ReverseDiff

for op in (:_get_device, :_get_device_type)
    @eval begin
        function MLDataDevices.$op(x::ReverseDiff.TrackedArray)
            return MLDataDevices.$op(ReverseDiff.value(x))
        end
        function MLDataDevices.$op(x::AbstractArray{<:ReverseDiff.TrackedReal})
            return MLDataDevices.$op(ReverseDiff.value.(x))
        end
    end
end

end
