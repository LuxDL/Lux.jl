module MLDataDevicesReverseDiffExt

using MLDataDevices: Internal
using ReverseDiff: ReverseDiff

for op in (:get_device, :get_device_type)
    @eval begin
        Internal.$(op)(x::ReverseDiff.TrackedArray) = Internal.$(op)(ReverseDiff.value(x))
        function Internal.$(op)(x::AbstractArray{<:ReverseDiff.TrackedReal})
            return Internal.$(op)(ReverseDiff.value.(x))
        end
    end
end

end
