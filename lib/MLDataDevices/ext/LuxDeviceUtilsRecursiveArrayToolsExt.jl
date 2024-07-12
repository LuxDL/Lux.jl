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

function LuxDeviceUtils._get_device(x::Union{VectorOfArray, DiffEqArray})
    length(x.u) == 0 && return nothing
    return mapreduce(LuxDeviceUtils.get_device, LuxDeviceUtils.__combine_devices, x.u)
end

function LuxDeviceUtils._get_device_type(x::Union{VectorOfArray, DiffEqArray})
    length(x.u) == 0 && return Nothing
    return mapreduce(LuxDeviceUtils._get_device_type, LuxDeviceUtils.__combine_devices, x.u)
end

end
