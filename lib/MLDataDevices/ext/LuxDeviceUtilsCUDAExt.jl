module LuxDeviceUtilsCUDAExt

using Adapt: Adapt
using CUDA: CUDA, CUSPARSE
using LuxDeviceUtils: LuxDeviceUtils, LuxCUDADevice, LuxCPUDevice
using Random: Random

function LuxDeviceUtils._with_device(::Type{LuxCUDADevice}, id::Int)
    id > length(CUDA.devices()) &&
        throw(ArgumentError("id = $id > length(CUDA.devices()) = $(length(CUDA.devices()))"))
    old_dev = CUDA.device()
    CUDA.device!(id - 1)
    device = LuxCUDADevice(CUDA.device())
    CUDA.device!(old_dev)
    return device
end

function LuxDeviceUtils._with_device(::Type{LuxCUDADevice}, ::Nothing)
    return LuxCUDADevice(nothing)
end

LuxDeviceUtils._get_device_id(dev::LuxCUDADevice) = CUDA.deviceid(dev.device) + 1

# Default RNG
LuxDeviceUtils.default_device_rng(::LuxCUDADevice) = CUDA.default_rng()

# Query Device from Array
LuxDeviceUtils.get_device(x::CUDA.AnyCuArray) = LuxCUDADevice(CUDA.device(x))

# Set Device
function LuxDeviceUtils.set_device!(::Type{LuxCUDADevice}, dev::CUDA.CuDevice)
    if !CUDA.functional()
        @warn "CUDA is not functional."
        return
    end
    CUDA.device!(dev)
    return
end
function LuxDeviceUtils.set_device!(::Type{LuxCUDADevice}, id::Int)
    if !CUDA.functional()
        @warn "CUDA is not functional."
        return
    end
    CUDA.device!(id - 1)
    return
end
function LuxDeviceUtils.set_device!(::Type{LuxCUDADevice}, ::Nothing, rank::Int)
    id = mod1(rank + 1, length(CUDA.devices()))
    return LuxDeviceUtils.set_device!(LuxCUDADevice, id)
end

# Device Transfer
## To GPU
Adapt.adapt_storage(::LuxCUDADevice{Nothing}, x) = CUDA.cu(x)
function Adapt.adapt_storage(to::LuxCUDADevice, x)
    old_dev = CUDA.device()  # remember the current device
    if !(x isa CUDA.AnyCuArray)
        CUDA.device!(to.device)
        x_new = CUDA.cu(x)
        CUDA.device!(old_dev)
        return x_new
    elseif CUDA.device(x) == to.device
        return x
    else
        CUDA.device!(to.device)
        x_new = copy(x)
        CUDA.device!(old_dev)
        return x_new
    end
end

Adapt.adapt_storage(::LuxCPUDevice, rng::CUDA.RNG) = Random.default_rng()

## To CPU
## FIXME: Use SparseArrays to preserve the sparsity
function Adapt.adapt_storage(::LuxCPUDevice, x::CUSPARSE.AbstractCuSparseMatrix)
    @warn "Currently we don't convert CUSPARSE matrices to CPU SparseArrays. Constructing \
           a dense matrix instead." maxlog=1
    return Adapt.adapt(Array, x)
end

end
