module LuxDeviceUtilsRecursiveArrayToolsExt

using Adapt: Adapt, adapt
using LuxDeviceUtils: LuxDeviceUtils, AbstractLuxDevice
using RecursiveArrayTools: VectorOfArray, DiffEqArray

# We want to preserve the structure
function Adapt.adapt_structure(to::AbstractLuxDevice, x::VectorOfArray)
    return VectorOfArray(map(Base.Fix1(adapt, to), x.u))
end

function Adapt.adapt_structure(to::AbstractLuxDevice, x::DiffEqArray)
    # Don't move the `time` to the GPU
    return DiffEqArray(map(Base.Fix1(adapt, to), x.u), x.t)
end

for op in (:_get_device, :_get_device_type)
    @eval function LuxDeviceUtils.$op(x::Union{VectorOfArray, DiffEqArray})
        length(x.u) == 0 && return $(op == :_get_device ? nothing : Nothing)
        return mapreduce(LuxDeviceUtils.$op, LuxDeviceUtils.__combine_devices, x.u)
    end
end

end
