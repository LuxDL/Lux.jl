module LuxDeviceUtilsZygoteExt

using Adapt, LuxDeviceUtils, Zygote

Adapt.adapt_structure(::LuxCPUAdaptor, x::Zygote.OneElement) = x

function Adapt.adapt_structure(to::LuxDeviceUtils.AbstractLuxDeviceAdaptor,
        x::Zygote.OneElement)
    return adapt(to, collect(x))
end

end
