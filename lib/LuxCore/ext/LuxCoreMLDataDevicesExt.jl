module LuxCoreMLDataDevicesExt

using LuxCore: LuxCore
using MLDataDevices: MLDataDevices

for (dev) in (:CPU, :CUDA, :AMDGPU, :Metal, :oneAPI)
    ldev = Symbol(dev, :Device)
    @eval function (::MLDataDevices.$(ldev))(NN::LuxCore.AbstractExplicitLayer)
        @warn "Lux layers are stateless and hence don't participate in device transfers. \
               Apply this function on the parameters and states generated using \
               `LuxCore.setup`."
        return NN
    end
end

end
