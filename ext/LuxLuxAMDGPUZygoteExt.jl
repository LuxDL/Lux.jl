module LuxLuxAMDGPUZygoteExt

if isdefined(Base, :get_extension)
    using Zygote
    using LuxAMDGPU
else
    using ..Zygote
    using ..LuxAMDGPU
end

using Adapt, Lux

Adapt.adapt_storage(::Lux.LuxAMDGPUAdaptor, x::Zygote.OneElement) = roc(collect(x))

end
