module ReverseDiffExt

using MLDataDevices: Internal
using ReverseDiff: ReverseDiff

for op in (:get_device, :get_device_type)
    @eval begin
        Internal.$(op)(x::ReverseDiff.TrackedArray) = Internal.$(op)(ReverseDiff.value(x))
        Internal.$(op)(x::AbstractArray{<:ReverseDiff.TrackedReal}) =
            Internal.$(op)(ReverseDiff.value.(x))
    end
end

end
