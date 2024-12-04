module LuxCoreMLDataDevicesExt

using Adapt: Adapt
using LuxCore: LuxCore
using MLDataDevices: MLDataDevices

MLDataDevices.isleaf(::LuxCore.AbstractLuxLayer) = true

for (dev) in (:CPU, :CUDA, :AMDGPU, :Metal, :oneAPI, :Reactant)
    ldev = Symbol(dev, :Device)
    @eval function Adapt.adapt_storage(::MLDataDevices.$(ldev), x::LuxCore.AbstractLuxLayer)
        @warn "Lux layers are stateless and hence don't participate in device transfers. \
               Apply this function on the parameters and states generated using \
               `LuxCore.setup`."
        return x
    end
end

end
