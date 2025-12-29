module MLDataDevicesExt

using Adapt: Adapt
using LuxCore: LuxCore, AbstractLuxLayer
using MLDataDevices: MLDataDevices, AbstractDevice

MLDataDevices.isleaf(::AbstractLuxLayer) = true

function Adapt.adapt_storage(::AbstractDevice, x::AbstractLuxLayer)
    @warn "Lux layers are stateless and hence don't participate in device transfers. \
           Apply this function on the parameters and states generated using \
           `LuxCore.setup`." maxlog = 1
    return x
end

end
