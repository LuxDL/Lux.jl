module Internal

using Preferences: load_preference
using Random: AbstractRNG
using UnrolledUtilities: unrolled_mapreduce

using ..MLDataDevices: MLDataDevices, AbstractDevice, CPUDevice, CUDADevice, AMDGPUDevice,
                       MetalDevice, oneAPIDevice, supported_gpu_backends, GPU_DEVICES,
                       loaded, functional

for dev in (CPUDevice, MetalDevice, oneAPIDevice)
    msg = "`device_id` is not applicable for `$dev`."
    @eval begin
        with_device(::Type{$dev}, ::Nothing) = $dev()
        function with_device(::Type{$dev}, device_id)
            @warn $(msg) maxlog=1
            return $dev()
        end
    end
end

for name in (:CPU, :CUDA, :AMDGPU, :Metal, :oneAPI)
    tpkg = name === :CPU ? "" : string(name)
    ldev = Symbol(name, :Device)
    @eval begin
        get_device_name(::Union{$ldev, Type{<:$ldev}}) = $(string(name))
        get_triggerpkg_name(::Union{$ldev, Type{<:$ldev}}) = $(tpkg)
    end
end

for T in (CPUDevice, CUDADevice{Nothing}, AMDGPUDevice{Nothing}, MetalDevice, oneAPIDevice)
    @eval get_device_id(::$(T)) = nothing
end

struct DeviceSelectionException <: Exception end

function Base.showerror(io::IO, ::DeviceSelectionException)
    return print(io, "DeviceSelectionException(No functional GPU device found!!)")
end

function get_gpu_device(; force_gpu_usage::Bool)
    backend = load_preference(MLDataDevices, "gpu_backend", nothing)

    # If backend set with preferences, use it
    if backend !== nothing
        allowed_backends = supported_gpu_backends()
        if backend ∉ allowed_backends
            @warn "`gpu_backend` preference is set to $backend, which is not a valid \
                    backend. Valid backends are $allowed_backends. Defaulting to automatic \
                    GPU Backend selection." maxlog=1
        else
            @debug "Using GPU backend set in preferences: $backend."
            idx = findfirst(isequal(backend), allowed_backends)
            device = GPU_DEVICES[idx]
            if !loaded(device)
                @warn "Trying to use backend: $(get_device_name(device)) but the trigger \
                       package $(get_triggerpkg_name(device)) is not loaded. Ignoring the \
                       Preferences backend!!! Please load the package and call this \
                       function again to respect the Preferences backend." maxlog=1
            else
                if functional(device)
                    @debug "Using GPU backend: $(get_device_name(device))."
                    return device
                else
                    @warn "GPU backend: $(get_device_name(device)) set via Preferences.jl \
                           is not functional. Defaulting to automatic GPU Backend \
                           selection." maxlog=1
                end
            end
        end
    end

    @debug "Running automatic GPU backend selection..."
    for device in GPU_DEVICES
        if loaded(device)
            @debug "Trying backend: $(get_device_name(device))."
            if functional(device)
                @debug "Using GPU backend: $(get_device_name(device))."
                return device
            end
            @debug "GPU backend: $(get_device_name(device)) is not functional."
        else
            @debug "Trigger package for backend ($(get_device_name(device))): \
                    $(get_triggerpkg_name(device)) not loaded."
        end
    end

    force_gpu_usage && throw(DeviceSelectionException())
    @warn """No functional GPU backend found! Defaulting to CPU.

             1. If no GPU is available, nothing needs to be done.
             2. If GPU is available, load the corresponding trigger package.
                 a. `CUDA.jl` and `cuDNN.jl` (or just `LuxCUDA.jl`) for  NVIDIA CUDA Support.
                 b. `AMDGPU.jl` for AMD GPU ROCM Support.
                 c. `Metal.jl` for Apple Metal GPU Support. (Experimental)
                 d. `oneAPI.jl` for Intel oneAPI GPU Support. (Experimental)""" maxlog=1
    return CPUDevice
end

special_aos(::AbstractArray) = false

recursive_array_eltype(::Type{T}) where {T} = !isbitstype(T) && !(T <: Number)

combine_devices(::Nothing, ::Nothing) = nothing
combine_devices(::Type{Nothing}, ::Type{Nothing}) = Nothing
combine_devices(::Nothing, dev::AbstractDevice) = dev
combine_devices(::Type{Nothing}, ::Type{T}) where {T <: AbstractDevice} = T
combine_devices(dev::AbstractDevice, ::Nothing) = dev
combine_devices(::Type{T}, ::Type{Nothing}) where {T <: AbstractDevice} = T
function combine_devices(dev1::AbstractDevice, dev2::AbstractDevice)
    dev1 == dev2 && return dev1
    throw(ArgumentError("Objects are on different devices: $(dev1) and $(dev2)."))
end
combine_devices(::Type{T}, ::Type{T}) where {T <: AbstractDevice} = T
function combine_devices(T1::Type{<:AbstractDevice}, T2::Type{<:AbstractDevice})
    throw(ArgumentError("Objects are on devices with different types: $(T1) and $(T2)."))
end

for op in (:get_device, :get_device_type)
    cpu_ret_val = op == :get_device ? CPUDevice() : CPUDevice

    @eval begin
        function $(op)(x::AbstractArray{T}) where {T}
            recursive_array_eltype(T) &&
                return mapreduce(MLDataDevices.$(op), combine_devices, x)
            if hasmethod(parent, Tuple{typeof(x)})
                parent_x = parent(x)
                parent_x === x && return $(cpu_ret_val)
                return $(op)(parent_x)
            end
            return $(cpu_ret_val)
        end

        function $(op)(x::Union{Tuple, NamedTuple})
            length(x) == 0 && return $(op == :get_device ? nothing : Nothing)
            return unrolled_mapreduce(MLDataDevices.$(op), combine_devices, values(x))
        end
    end

    for T in (Number, AbstractRNG, Val, Symbol, String, Nothing)
        @eval $(op)(::$(T)) = $(op == :get_device ? nothing : Nothing)
    end
end

end
