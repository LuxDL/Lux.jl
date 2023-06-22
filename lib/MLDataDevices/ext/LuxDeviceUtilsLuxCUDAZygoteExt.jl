module LuxDeviceUtilsLuxCUDAZygoteExt

if isdefined(Base, :get_extension)
    using Zygote
    using LuxCUDA
else
    using ..Zygote
    using ..LuxCUDA
end

using Adapt, LuxDeviceUtils

Adapt.adapt_storage(::LuxCUDAAdaptor, x::Zygote.OneElement) = cu(collect(x))

end
