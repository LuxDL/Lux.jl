module LuxDeviceUtilsRecursiveArrayToolsExt

using Adapt, LuxDeviceUtils, RecursiveArrayTools

# We want to preserve the structure
function Adapt.adapt_structure(to::LuxDeviceUtils.AbstractLuxDeviceAdaptor,
        x::VectorOfArray)
    return VectorOfArray(map(Base.Fix1(adapt, to), x.u))
end

function Adapt.adapt_structure(to::LuxDeviceUtils.AbstractLuxDeviceAdaptor,
        x::DiffEqArray)
    # Don't move the `time` to the GPU
    return DiffEqArray(map(Base.Fix1(adapt, to), x.u), x.t)
end

end
