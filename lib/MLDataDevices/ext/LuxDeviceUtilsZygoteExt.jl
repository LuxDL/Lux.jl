module LuxDeviceUtilsZygoteExt

isdefined(Base, :get_extension) ? (using Zygote) : (using ..Zygote)

using Adapt, LuxDeviceUtils

Adapt.adapt_structure(::LuxCPUAdaptor, x::Zygote.OneElement) = x

function Adapt.adapt_structure(to::LuxDeviceUtils.AbstractLuxDeviceAdaptor,
    x::Zygote.OneElement)
    return adapt(to, collect(x))
end

end
