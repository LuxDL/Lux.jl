module LuxDeviceUtilsZygoteExt

isdefined(Base, :get_extension) ? (using Zygote) : (using ..Zygote)

using Adapt, LuxDeviceUtils

Adapt.adapt_storage(::LuxCPUAdaptor, x::Zygote.OneElement) = x

function Adapt.adapt_structure(to::LuxDeviceUtils.AbstractLuxDeviceAdaptor,
    x::Zygote.OneElement)
    return Adapt.adapt_structure(to, collect(x))
end

end
