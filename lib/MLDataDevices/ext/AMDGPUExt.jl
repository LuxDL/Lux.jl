module AMDGPUExt

using Adapt: Adapt
using AMDGPU: AMDGPU, ROCArray
using MLDataDevices: MLDataDevices, Internal, AMDGPUDevice, CPUDevice, reset_gpu_device!
using Random: Random

__init__() = reset_gpu_device!()

# This code used to be in `LuxAMDGPU.jl`, but we no longer need that package.
const USE_AMD_GPU = Ref{Union{Nothing,Bool}}(nothing)

function check_use_amdgpu!()
    USE_AMD_GPU[] === nothing || return nothing

    USE_AMD_GPU[] = AMDGPU.functional()
    if USE_AMD_GPU[] && !AMDGPU.functional(:MIOpen)
        @warn "MIOpen is not functional in AMDGPU.jl, some functionality will not be \
               available." maxlog = 1
    end
    return nothing
end

MLDataDevices.loaded(::Union{AMDGPUDevice,<:Type{AMDGPUDevice}}) = true
function MLDataDevices.functional(::Union{AMDGPUDevice,<:Type{AMDGPUDevice}})::Bool
    check_use_amdgpu!()
    return USE_AMD_GPU[]
end

Internal.with_device(::Type{AMDGPUDevice}, ::Nothing) = AMDGPUDevice(nothing)
function Internal.with_device(::Type{AMDGPUDevice}, id::Integer)
    id > length(AMDGPU.devices()) && throw(
        ArgumentError("id = $id > length(AMDGPU.devices()) = $(length(AMDGPU.devices()))"),
    )
    old_dev = AMDGPU.device()
    AMDGPU.device!(AMDGPU.devices()[id])
    device = AMDGPUDevice(AMDGPU.device())
    AMDGPU.device!(old_dev)
    return device
end

Internal.get_device_id(dev::AMDGPUDevice) = AMDGPU.device_id(dev.device)

# Default RNG
MLDataDevices.default_device_rng(::AMDGPUDevice) = AMDGPU.rocrand_rng()

# Query Device from Array
function Internal.get_device(x::AMDGPU.AnyROCArray)
    parent_x = parent(x)
    parent_x === x && return AMDGPUDevice(AMDGPU.device(x))
    return Internal.get_device(parent_x)
end
Internal.get_device(::AMDGPU.rocRAND.RNG) = AMDGPUDevice(AMDGPU.device())

Internal.get_device_type(::AMDGPU.AnyROCArray) = AMDGPUDevice
Internal.get_device_type(::AMDGPU.rocRAND.RNG) = AMDGPUDevice

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

# unsafe_free!
function Internal.unsafe_free_internal!(::Type{AMDGPUDevice}, x::AbstractArray)
    if applicable(AMDGPU.unsafe_free!, x)
        AMDGPU.unsafe_free!(x)
    else
        @warn "AMDGPU.unsafe_free! is not defined for $(typeof(x))." maxlog = 1
    end
    return nothing
end

# Device Transfer
function amdgpu_array_adapt(::Type{T}, x) where {T}
    return Internal.array_adapt(AMDGPU.roc, ROCArray, T, x)
end

function Adapt.adapt_storage(::AMDGPUDevice{D,Missing}, x::AbstractArray) where {D}
    MLDataDevices.get_device_type(x) <: AMDGPUDevice && return x
    return amdgpu_array_adapt(Missing, x)
end

function Adapt.adapt_storage(::AMDGPUDevice{D,Nothing}, x::AbstractArray) where {D}
    MLDataDevices.get_device_type(x) <: AMDGPUDevice && return x
    return amdgpu_array_adapt(Nothing, x)
end

function Adapt.adapt_storage(
    ::AMDGPUDevice{D,T}, x::AbstractArray{ET}
) where {D,T<:AbstractFloat,ET<:Number}
    MLDataDevices.get_device_type(x) <: AMDGPUDevice && ET == T && return x
    return amdgpu_array_adapt(T, x)
end

function Adapt.adapt_storage(to::AMDGPUDevice{D,E}, x::AbstractArray) where {D,E}
    old_dev = AMDGPU.device()  # remember the current device
    dev = MLDataDevices.get_device(x)
    if !(dev isa AMDGPUDevice)
        AMDGPU.device!(to.device)
        x_new = amdgpu_array_adapt(to, x)
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
