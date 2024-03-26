module LuxDeviceUtilsAMDGPUExt

using Adapt: Adapt
using AMDGPU: AMDGPU
using LuxDeviceUtils: LuxDeviceUtils, LuxAMDGPUAdaptor, LuxAMDGPUDevice, LuxCPUAdaptor
using Random: Random, AbstractRNG

function LuxDeviceUtils._with_device(::Type{LuxAMDGPUDevice}, ::Nothing)
    return LuxAMDGPUDevice(nothing)
end
function LuxDeviceUtils._with_device(::Type{LuxAMDGPUDevice}, id::Int)
    id > length(AMDGPU.devices()) &&
        throw(ArgumentError("id = $id > length(AMDGPU.devices()) = $(length(AMDGPU.devices()))"))
    old_dev = AMDGPU.device()
    AMDGPU.device!(AMDGPU.devices()[id])
    device = LuxAMDGPUDevice(AMDGPU.device())
    AMDGPU.device!(old_dev)
    return device
end

LuxDeviceUtils._get_device_id(dev::LuxAMDGPUDevice) = AMDGPU.device_id(dev.device)

# Default RNG
LuxDeviceUtils.default_device_rng(::LuxAMDGPUDevice) = AMDGPU.rocrand_rng()

# Query Device from Array
LuxDeviceUtils.get_device(x::AMDGPU.AnyROCArray) = LuxAMDGPUDevice(AMDGPU.device(x))

# Device Transfer
## To GPU
Adapt.adapt_storage(::LuxAMDGPUAdaptor{Nothing}, x) = AMDGPU.roc(x)
function Adapt.adapt_storage(to::LuxAMDGPUAdaptor, x)
    old_dev = AMDGPU.device()  # remember the current device
    if !(x isa AMDGPU.AnyROCArray)
        AMDGPU.device!(to.device)
        x_new = AMDGPU.roc(x)
        AMDGPU.device!(old_dev)
        return x_new
    elseif AMDGPU.device_id(AMDGPU.device(x)) == AMDGPU.device_id(to.device)
        return x
    else
        AMDGPU.device!(to.device)
        x_new = copy(x)
        AMDGPU.device!(old_dev)
        return x_new
    end
end
Adapt.adapt_storage(::LuxAMDGPUAdaptor{Nothing}, rng::AbstractRNG) = rng
Adapt.adapt_storage(::LuxAMDGPUAdaptor, rng::AbstractRNG) = rng
function Adapt.adapt_storage(::LuxAMDGPUAdaptor{Nothing}, rng::Random.TaskLocalRNG)
    return AMDGPU.rocrand_rng()
end
Adapt.adapt_storage(::LuxAMDGPUAdaptor, rng::Random.TaskLocalRNG) = AMDGPU.rocrand_rng()

Adapt.adapt_storage(::LuxCPUAdaptor, rng::AMDGPU.rocRAND.RNG) = Random.default_rng()

end
