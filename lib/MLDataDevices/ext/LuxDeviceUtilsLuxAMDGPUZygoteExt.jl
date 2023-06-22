module LuxDeviceUtilsLuxAMDGPUZygoteExt

if isdefined(Base, :get_extension)
    using Zygote
    using LuxAMDGPU
else
    using ..Zygote
    using ..LuxAMDGPU
end

using Adapt, LuxDeviceUtils

Adapt.adapt_storage(::LuxAMDGPUAdaptor, x::Zygote.OneElement) = roc(collect(x))

end
