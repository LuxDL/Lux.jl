module DeviceUtilsRecursiveArrayToolsExt

using Adapt: Adapt, adapt
using DeviceUtils: DeviceUtils, AbstractDevice
using RecursiveArrayTools: VectorOfArray, DiffEqArray

# We want to preserve the structure
function Adapt.adapt_structure(to::AbstractDevice, x::VectorOfArray)
    return VectorOfArray(map(Base.Fix1(adapt, to), x.u))
end

function Adapt.adapt_structure(to::AbstractDevice, x::DiffEqArray)
    # Don't move the `time` to the GPU
    return DiffEqArray(map(Base.Fix1(adapt, to), x.u), x.t)
end

for op in (:_get_device, :_get_device_type)
    @eval function DeviceUtils.$op(x::Union{VectorOfArray, DiffEqArray})
        length(x.u) == 0 && return $(op == :_get_device ? nothing : Nothing)
        return mapreduce(DeviceUtils.$op, DeviceUtils.__combine_devices, x.u)
    end
end

end
