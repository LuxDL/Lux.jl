module LuxDeviceUtilsZygoteExt

using Adapt: Adapt
using LuxDeviceUtils: AbstractLuxDeviceAdaptor, LuxCPUAdaptor
using Zygote: OneElement

Adapt.adapt_structure(::LuxCPUAdaptor, x::OneElement) = x

function Adapt.adapt_structure(to::AbstractLuxDeviceAdaptor, x::OneElement)
    return Adapt.adapt(to, collect(x))
end

end
