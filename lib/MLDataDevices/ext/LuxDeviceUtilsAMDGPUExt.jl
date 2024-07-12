module LuxDeviceUtilsAMDGPUExt

using Adapt: Adapt
using AMDGPU: AMDGPU
using LuxDeviceUtils: LuxDeviceUtils, LuxAMDGPUDevice, LuxCPUDevice, reset_gpu_device!
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

LuxDeviceUtils.loaded(::Union{LuxAMDGPUDevice, <:Type{LuxAMDGPUDevice}}) = true
function LuxDeviceUtils.functional(::Union{LuxAMDGPUDevice, <:Type{LuxAMDGPUDevice}})::Bool
    _check_use_amdgpu!()
    return USE_AMD_GPU[]
end

function LuxDeviceUtils._with_device(::Type{LuxAMDGPUDevice}, ::Nothing)
    return LuxAMDGPUDevice(nothing)
end
function LuxDeviceUtils._with_device(::Type{LuxAMDGPUDevice}, id::Integer)
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
function LuxDeviceUtils.get_device(x::AMDGPU.AnyROCArray)
    parent_x = parent(x)
    parent_x === x && return LuxAMDGPUDevice(AMDGPU.device(x))
    return LuxDeviceUtils.get_device(parent_x)
end

LuxDeviceUtils._get_device_type(::AMDGPU.AnyROCArray) = LuxAMDGPUDevice

# Set Device
function LuxDeviceUtils.set_device!(::Type{LuxAMDGPUDevice}, dev::AMDGPU.HIPDevice)
    return AMDGPU.device!(dev)
end
function LuxDeviceUtils.set_device!(::Type{LuxAMDGPUDevice}, id::Integer)
    return LuxDeviceUtils.set_device!(LuxAMDGPUDevice, AMDGPU.devices()[id])
end
function LuxDeviceUtils.set_device!(::Type{LuxAMDGPUDevice}, ::Nothing, rank::Integer)
    id = mod1(rank + 1, length(AMDGPU.devices()))
    return LuxDeviceUtils.set_device!(LuxAMDGPUDevice, id)
end

# Device Transfer
## To GPU
Adapt.adapt_storage(::LuxAMDGPUDevice{Nothing}, x::AbstractArray) = AMDGPU.roc(x)
function Adapt.adapt_storage(to::LuxAMDGPUDevice, x::AbstractArray)
    old_dev = AMDGPU.device()  # remember the current device
    dev = LuxDeviceUtils.get_device(x)
    if !(dev isa LuxAMDGPUDevice)
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

Adapt.adapt_storage(::LuxCPUDevice, rng::AMDGPU.rocRAND.RNG) = Random.default_rng()

end
