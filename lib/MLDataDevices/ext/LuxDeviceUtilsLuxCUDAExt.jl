module LuxDeviceUtilsLuxCUDAExt

using LuxCUDA, LuxDeviceUtils, Random
import Adapt: adapt_storage, adapt

__init__() = reset_gpu_device!()

LuxDeviceUtils.__is_loaded(::Union{LuxCUDADevice, Type{<:LuxCUDADevice}}) = true
function LuxDeviceUtils.__is_functional(::Union{LuxCUDADevice, Type{<:LuxCUDADevice}})
    return LuxCUDA.functional()
end

function LuxDeviceUtils._with_device(::Type{LuxCUDADevice}, ::Nothing)
    return LuxCUDADevice(nothing)
end
function LuxDeviceUtils._with_device(::Type{LuxCUDADevice}, id::Int)
    id > length(CUDA.devices()) &&
        throw(ArgumentError("id = $id > length(CUDA.devices()) = $(length(CUDA.devices()))"))
    old_dev = CUDA.device()
    CUDA.device!(id - 1)
    device = LuxCUDADevice(CUDA.device())
    CUDA.device!(old_dev)
    return device
end

LuxDeviceUtils._get_device_id(dev::LuxCUDADevice) = CUDA.deviceid(dev.device) + 1

# Default RNG
LuxDeviceUtils.default_device_rng(::LuxCUDADevice) = CUDA.default_rng()

# Query Device from Array
LuxDeviceUtils.get_device(::CUDA.AnyCuArray) = LuxCUDADevice()

# Device Transfer
## To GPU
adapt_storage(::LuxCUDAAdaptor{Nothing}, x) = cu(x)
function adapt_storage(to::LuxCUDAAdaptor, x)
    old_dev = CUDA.device()  # remember the current device
    if !(x isa CUDA.AnyCuArray)
        CUDA.device!(to.device)
        x_new = cu(x)
        CUDA.device!(old_dev)
        return x_new
    elseif CUDA.device(x).handle == to.device.handle
        return x
    else
        CUDA.device!(to.device)
        x_new = copy(x)
        CUDA.device!(old_dev)
        return x_new
    end
end
adapt_storage(::LuxCUDAAdaptor, rng::AbstractRNG) = rng
adapt_storage(::LuxCUDAAdaptor, rng::Random.TaskLocalRNG) = CUDA.default_rng()

adapt_storage(::LuxCPUAdaptor, rng::CUDA.RNG) = Random.default_rng()

## To CPU
adapt_storage(::LuxCPUAdaptor, x::CUSPARSE.AbstractCuSparseMatrix) = adapt(Array, x)

end
