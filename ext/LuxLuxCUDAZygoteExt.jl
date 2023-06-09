module LuxLuxCUDAZygoteExt

if isdefined(Base, :get_extension)
    using Zygote
    using LuxCUDA
else
    using ..Zygote
    using ..LuxCUDA
end

using Adapt, Lux

Adapt.adapt_storage(::Lux.LuxCUDAAdaptor, x::Zygote.OneElement) = CUDA.cu(collect(x))

end
