module DeviceUtilsAMDGPUExt

using Adapt: Adapt
using AMDGPU: AMDGPU
using MLDataDevices: MLDataDevices, AMDGPUDevice, CPUDevice, reset_gpu_device!
using Random: Random

__init__() = reset_gpu_device!()

# This code used to be in `LuxAMDGPU.jl`, but we no longer need that package.
const USE_AMD_GPU = Ref{Union{Nothing, Bool}}(nothing)

function _check_use_amdgpu!()
    USE_AMD_GPU[] === nothing || return

    USE_AMD_GPU[] = AMDGPU.functional()
    if USE_AMD_GPU[] && !AMDGPU.functional(:MIOpen)
        @warn "MIOpen is not functional in AMDGPU.jl, some functionality will not be \
               available." maxlog=1
    end
    return
end

MLDataDevices.loaded(::Union{AMDGPUDevice, <:Type{AMDGPUDevice}}) = true
function MLDataDevices.functional(::Union{AMDGPUDevice, <:Type{AMDGPUDevice}})::Bool
    _check_use_amdgpu!()
    return USE_AMD_GPU[]
end

function MLDataDevices._with_device(::Type{AMDGPUDevice}, ::Nothing)
    return AMDGPUDevice(nothing)
end
function MLDataDevices._with_device(::Type{AMDGPUDevice}, id::Integer)
    id > length(AMDGPU.devices()) &&
        throw(ArgumentError("id = $id > length(AMDGPU.devices()) = $(length(AMDGPU.devices()))"))
    old_dev = AMDGPU.device()
    AMDGPU.device!(AMDGPU.devices()[id])
    device = AMDGPUDevice(AMDGPU.device())
    AMDGPU.device!(old_dev)
    return device
end

MLDataDevices._get_device_id(dev::AMDGPUDevice) = AMDGPU.device_id(dev.device)

# Default RNG
MLDataDevices.default_device_rng(::AMDGPUDevice) = AMDGPU.rocrand_rng()

# Query Device from Array
function MLDataDevices._get_device(x::AMDGPU.AnyROCArray)
    parent_x = parent(x)
    parent_x === x && return AMDGPUDevice(AMDGPU.device(x))
    return MLDataDevices._get_device(parent_x)
end

MLDataDevices._get_device_type(::AMDGPU.AnyROCArray) = AMDGPUDevice

# Set Device
function MLDataDevices.set_device!(::Type{AMDGPUDevice}, dev::AMDGPU.HIPDevice)
    return AMDGPU.device!(dev)
end
function MLDataDevices.set_device!(::Type{AMDGPUDevice}, id::Integer)
    return MLDataDevices.set_device!(AMDGPUDevice, AMDGPU.devices()[id])
end
function MLDataDevices.set_device!(::Type{AMDGPUDevice}, ::Nothing, rank::Integer)
    id = mod1(rank + 1, length(AMDGPU.devices()))
    return MLDataDevices.set_device!(AMDGPUDevice, id)
end

# Device Transfer
## To GPU
Adapt.adapt_storage(::AMDGPUDevice{Nothing}, x::AbstractArray) = AMDGPU.roc(x)
function Adapt.adapt_storage(to::AMDGPUDevice, x::AbstractArray)
    old_dev = AMDGPU.device()  # remember the current device
    dev = MLDataDevices.get_device(x)
    if !(dev isa AMDGPUDevice)
        AMDGPU.device!(to.device)
        x_new = AMDGPU.roc(x)
        AMDGPU.device!(old_dev)
        return x_new
    elseif AMDGPU.device_id(dev.device) == AMDGPU.device_id(to.device)
        return x
    else
        AMDGPU.device!(to.device)
        x_new = copy(x)
        AMDGPU.device!(old_dev)
        return x_new
    end
end

Adapt.adapt_storage(::CPUDevice, rng::AMDGPU.rocRAND.RNG) = Random.default_rng()

end
