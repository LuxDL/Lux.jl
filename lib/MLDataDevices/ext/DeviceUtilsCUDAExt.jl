module DeviceUtilsCUDAExt

using Adapt: Adapt
using CUDA: CUDA
using CUDA.CUSPARSE: AbstractCuSparseMatrix, AbstractCuSparseVector
using DeviceUtils: DeviceUtils, CUDADevice, CPUDevice
using Random: Random

function DeviceUtils._with_device(::Type{CUDADevice}, id::Integer)
    id > length(CUDA.devices()) &&
        throw(ArgumentError("id = $id > length(CUDA.devices()) = $(length(CUDA.devices()))"))
    old_dev = CUDA.device()
    CUDA.device!(id - 1)
    device = CUDADevice(CUDA.device())
    CUDA.device!(old_dev)
    return device
end

function DeviceUtils._with_device(::Type{CUDADevice}, ::Nothing)
    return CUDADevice(nothing)
end

DeviceUtils._get_device_id(dev::CUDADevice) = CUDA.deviceid(dev.device) + 1

# Default RNG
DeviceUtils.default_device_rng(::CUDADevice) = CUDA.default_rng()

# Query Device from Array
function DeviceUtils._get_device(x::CUDA.AnyCuArray)
    parent_x = parent(x)
    parent_x === x && return CUDADevice(CUDA.device(x))
    return DeviceUtils.get_device(parent_x)
end
function DeviceUtils._get_device(x::CUDA.CUSPARSE.AbstractCuSparseArray)
    return CUDADevice(CUDA.device(x.nzVal))
end

function DeviceUtils._get_device_type(::Union{
        <:CUDA.AnyCuArray, <:CUDA.CUSPARSE.AbstractCuSparseArray})
    return CUDADevice
end

# Set Device
function DeviceUtils.set_device!(::Type{CUDADevice}, dev::CUDA.CuDevice)
    return CUDA.device!(dev)
end
function DeviceUtils.set_device!(::Type{CUDADevice}, id::Integer)
    return DeviceUtils.set_device!(CUDADevice, collect(CUDA.devices())[id])
end
function DeviceUtils.set_device!(::Type{CUDADevice}, ::Nothing, rank::Integer)
    id = mod1(rank + 1, length(CUDA.devices()))
    return DeviceUtils.set_device!(CUDADevice, id)
end

# Device Transfer
Adapt.adapt_storage(::CUDADevice{Nothing}, x::AbstractArray) = CUDA.cu(x)
function Adapt.adapt_storage(to::CUDADevice, x::AbstractArray)
    old_dev = CUDA.device()  # remember the current device
    dev = DeviceUtils.get_device(x)
    if !(dev isa CUDADevice)
        CUDA.device!(to.device)
        x_new = CUDA.cu(x)
        CUDA.device!(old_dev)
        return x_new
    elseif dev.device == to.device
        return x
    else
        CUDA.device!(to.device)
        x_new = copy(x)
        CUDA.device!(old_dev)
        return x_new
    end
end

Adapt.adapt_storage(::CPUDevice, rng::CUDA.RNG) = Random.default_rng()

# Defining as extensions seems to case precompilation errors
@static if isdefined(CUDA.CUSPARSE, :SparseArrays)
    function Adapt.adapt_storage(::CPUDevice, x::AbstractCuSparseMatrix)
        return CUDA.CUSPARSE.SparseArrays.SparseMatrixCSC(x)
    end
    function Adapt.adapt_storage(::CPUDevice, x::AbstractCuSparseVector)
        return CUDA.CUSPARSE.SparseArrays.SparseVector(x)
    end
else
    @warn "CUDA.CUSPARSE seems to have removed SparseArrays as a dependency. Please open \
           an issue in DeviceUtils.jl repository."
end

end
