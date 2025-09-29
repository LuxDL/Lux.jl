module MLDataDevicesCUDAExt

using Adapt: Adapt
using CUDA: CUDA
using CUDA.CUSPARSE: AbstractCuSparseMatrix, AbstractCuSparseVector, AbstractCuSparseArray
using MLDataDevices: MLDataDevices, Internal, CUDADevice, CPUDevice
using Random: Random

Internal.with_device(::Type{CUDADevice}, ::Nothing) = CUDADevice()
function Internal.with_device(::Type{CUDADevice}, id::Integer)
    id > length(CUDA.devices()) && throw(
        ArgumentError("id = $id > length(CUDA.devices()) = $(length(CUDA.devices()))")
    )
    old_dev = CUDA.device()
    CUDA.device!(id - 1)
    device = CUDADevice(CUDA.device())
    CUDA.device!(old_dev)
    return device
end

Internal.get_device_id(dev::CUDADevice) = CUDA.deviceid(dev.device) + 1

# Default RNG
MLDataDevices.default_device_rng(::CUDADevice) = CUDA.default_rng()

# Query Device from Array
function Internal.get_device(x::CUDA.AnyCuArray)
    parent_x = parent(x)
    parent_x === x && return CUDADevice(CUDA.device(x))
    return MLDataDevices.get_device(parent_x)
end
Internal.get_device(x::AbstractCuSparseArray) = CUDADevice(CUDA.device(x.nzVal))
Internal.get_device(::CUDA.RNG) = CUDADevice(CUDA.device())
Internal.get_device(::CUDA.CURAND.RNG) = CUDADevice(CUDA.device())

Internal.get_device_type(::Union{<:CUDA.AnyCuArray,<:AbstractCuSparseArray}) = CUDADevice
Internal.get_device_type(::CUDA.RNG) = CUDADevice
Internal.get_device_type(::CUDA.CURAND.RNG) = CUDADevice

# Set Device
MLDataDevices.set_device!(::Type{CUDADevice}, dev::CUDA.CuDevice) = CUDA.device!(dev)
function MLDataDevices.set_device!(::Type{CUDADevice}, id::Integer)
    return MLDataDevices.set_device!(CUDADevice, collect(CUDA.devices())[id])
end
function MLDataDevices.set_device!(::Type{CUDADevice}, ::Nothing, rank::Integer)
    id = mod1(rank + 1, length(CUDA.devices()))
    return MLDataDevices.set_device!(CUDADevice, id)
end

# unsafe_free!
function Internal.unsafe_free_internal!(::Type{CUDADevice}, x::AbstractArray)
    if applicable(CUDA.unsafe_free!, x)
        CUDA.unsafe_free!(x)
    else
        @warn "CUDA.unsafe_free! is not defined for $(typeof(x))." maxlog = 1
    end
    return nothing
end

# Device Transfer
function Adapt.adapt_storage(dev::CUDADevice{Nothing}, x::AbstractArray)
    MLDataDevices.get_device_type(x) <: CUDADevice && return x
    if dev.eltype === missing
        # Use CUDA.cu which does automatic eltype conversion (FP64 -> FP32)
        return CUDA.cu(x)
    elseif dev.eltype === nothing
        # Preserve the original eltype
        return CuArray(x)
    else
        # Convert to specified eltype first, then move to GPU
        x_converted = MLDataDevices._maybe_convert_eltype(dev, x)
        return CuArray(x_converted)
    end
end

function Adapt.adapt_storage(to::CUDADevice, x::AbstractArray)
    old_dev = CUDA.device()  # remember the current device
    dev = MLDataDevices.get_device(x)
    if !(dev isa CUDADevice)
        CUDA.device!(to.device)
        if to.eltype === missing
            # Use CUDA.cu which does automatic eltype conversion (FP64 -> FP32)
            x_new = CUDA.cu(x)
        elseif to.eltype === nothing
            # Preserve the original eltype
            x_new = CuArray(x)
        else
            # Convert to specified eltype first, then move to GPU
            x_converted = MLDataDevices._maybe_convert_eltype(to, x)
            x_new = CuArray(x_converted)
        end
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
           an issue in MLDataDevices.jl repository."
end

end
