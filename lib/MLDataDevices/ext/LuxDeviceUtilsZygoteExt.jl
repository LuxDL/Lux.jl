module LuxDeviceUtilsZygoteExt

isdefined(Base, :get_extension) ? (using Zygote) : (using ..Zygote)

using Adapt, LuxDeviceUtils

Adapt.adapt_storage(::LuxCPUAdaptor, x::Zygote.OneElement) = x

end
