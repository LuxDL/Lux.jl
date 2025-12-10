module Internal

using Functors: fleaves
using Preferences: load_preference
using Random: AbstractRNG

using ..MLDataDevices:
    MLDataDevices,
    AbstractDevice,
    CPUDevice,
    CUDADevice,
    AMDGPUDevice,
    MetalDevice,
    oneAPIDevice,
    OpenCLDevice,
    ReactantDevice,
    UnknownDevice,
    supported_gpu_backends,
    GPU_DEVICES,
    loaded,
    functional

for dev in (CPUDevice, MetalDevice, oneAPIDevice, OpenCLDevice)
    msg = "`device_id` is not applicable for `$dev`."
    @eval begin
        with_device(::Type{$dev}, ::Nothing) = $dev()
        function with_device(::Type{$dev}, device_id)
            @warn $(msg) maxlog = 1
            return $dev()
        end
    end
end

for name in (:CPU, :CUDA, :AMDGPU, :Metal, :oneAPI, :OpenCL)
    tpkg = name === :CPU ? "" : string(name)
    ldev = Symbol(name, :Device)
    @eval begin
        get_device_name(::Union{$ldev,Type{<:$ldev}}) = $(string(name))
        get_triggerpkg_name(::Union{$ldev,Type{<:$ldev}}) = $(tpkg)
    end
end
get_device_name(::ReactantDevice) = "Reactant"
get_triggerpkg_name(::ReactantDevice) = "Reactant"

for T in (
    CPUDevice,
    CUDADevice{Nothing},
    AMDGPUDevice{Nothing},
    MetalDevice,
    oneAPIDevice,
    OpenCLDevice,
    ReactantDevice,
)
    @eval get_device_id(::$(T)) = nothing
end

struct DeviceSelectionException <: Exception
    dev::String
end

function Base.showerror(io::IO, d::DeviceSelectionException)
    return print(io, "DeviceSelectionException: No functional $(d.dev) device found!")
end

function get_gpu_device(; force::Bool)
    backend = load_preference(MLDataDevices, "gpu_backend", nothing)

    # If backend set with preferences, use it
    if backend !== nothing
        allowed_backends = supported_gpu_backends()
        if backend ∉ allowed_backends
            @warn "`gpu_backend` preference is set to $backend, which is not a valid \
                    backend. Valid backends are $allowed_backends. Defaulting to automatic \
                    GPU Backend selection." maxlog = 1
        else
            @debug "Using GPU backend set in preferences: $backend."
            idx = findfirst(isequal(backend), allowed_backends)
            device = GPU_DEVICES[idx]
            if !loaded(device)
                @warn "Trying to use backend: $(get_device_name(device)) but the trigger \
                       package $(get_triggerpkg_name(device)) is not loaded. Ignoring the \
                       Preferences backend!!! Please load the package and call this \
                       function again to respect the Preferences backend." maxlog = 1
            else
                if functional(device)
                    @debug "Using GPU backend: $(get_device_name(device))."
                    return device
                else
                    @warn "GPU backend: $(get_device_name(device)) set via Preferences.jl \
                           is not functional. Defaulting to automatic GPU Backend \
                           selection." maxlog = 1
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

    force && throw(DeviceSelectionException("GPU"))
    Base.get_bool_env("MLDATADEVICES_SILENCE_WARN_NO_GPU", false) ||
        @warn """No functional GPU backend found! Defaulting to CPU.

    1. If no GPU is available, nothing needs to be done. Set `MLDATADEVICES_SILENCE_WARN_NO_GPU=1` to silence this warning.
    2. If GPU is available, load the corresponding trigger package.
        a. `CUDA.jl` and `cuDNN.jl` (or just `LuxCUDA.jl`) for  NVIDIA CUDA Support.
        b. `AMDGPU.jl` for AMD GPU ROCM Support.
        c. `Metal.jl` for Apple Metal GPU Support. (Experimental)
        d. `oneAPI.jl` for Intel oneAPI GPU Support. (Experimental)
        e. `OpenCL.jl` for OpenCL support. (Experimental)""" maxlog = 1
    return CPUDevice
end

special_aos(::AbstractArray) = false

combine_devices(::Nothing, ::Nothing) = nothing
combine_devices(::Nothing, dev::AbstractDevice) = dev
combine_devices(dev::AbstractDevice, ::Nothing) = dev
function combine_devices(dev1::AbstractDevice, dev2::AbstractDevice)
    dev1 == dev2 && return dev1
    dev1 isa UnknownDevice && return dev2
    dev2 isa UnknownDevice && return dev1
    throw(ArgumentError("Objects are on different devices: $(dev1) and $(dev2)."))
end

combine_devices(::Type{Nothing}, ::Type{Nothing}) = Nothing
combine_devices(::Type{T}, ::Type{T}) where {T<:AbstractDevice} = T
combine_devices(::Type{T}, ::Type{Nothing}) where {T<:AbstractDevice} = T
combine_devices(::Type{T}, ::Type{UnknownDevice}) where {T<:AbstractDevice} = T
combine_devices(::Type{Nothing}, ::Type{T}) where {T<:AbstractDevice} = T
combine_devices(::Type{UnknownDevice}, ::Type{T}) where {T<:AbstractDevice} = T
combine_devices(::Type{UnknownDevice}, ::Type{UnknownDevice}) = UnknownDevice
function combine_devices(T1::Type{<:AbstractDevice}, T2::Type{<:AbstractDevice})
    throw(ArgumentError("Objects are on devices with different types: $(T1) and $(T2)."))
end

# Special cases for ReactantDevice
combine_devices(dev::ReactantDevice, ::AbstractDevice) = dev
combine_devices(::AbstractDevice, dev::ReactantDevice) = dev
function combine_devices(dev1::ReactantDevice, dev2::ReactantDevice)
    if dev1 == dev2
        # `merge(...)` of `IdDict` constructs a `Dict`
        if dev1.sharding isa IdDict
            sharding = dev1.sharding
            if dev2.sharding isa IdDict
                for (k, v) in dev2.sharding
                    sharding[k] = v
                end
            end
        elseif dev2.sharding isa IdDict
            sharding = dev2.sharding
        else
            sharding = missing
        end

        client = dev1.client === missing ? dev2.client : dev1.client
        device = dev1.device === missing ? dev2.device : dev1.device

        return ReactantDevice(client, device, sharding)
    end
    throw(ArgumentError("Objects are on different devices: $(dev1) and $(dev2)."))
end
combine_devices(::Type{ReactantDevice}, ::Type{UnknownDevice}) = ReactantDevice
combine_devices(::Type{UnknownDevice}, ::Type{ReactantDevice}) = ReactantDevice
function combine_devices(::Type{ReactantDevice}, ::Type{T}) where {T<:AbstractDevice}
    return ReactantDevice
end
function combine_devices(::Type{T}, ::Type{ReactantDevice}) where {T<:AbstractDevice}
    return ReactantDevice
end
combine_devices(::Type{ReactantDevice}, ::Type{ReactantDevice}) = ReactantDevice

for op in (:get_device, :get_device_type)
    cpu_ret_val = op == :get_device ? CPUDevice() : CPUDevice
    unknown_ret_val = op == :get_device ? UnknownDevice() : UnknownDevice
    all_not_assigned_msg = "AbstractArray has all undefined references. Giving up, \
                            returning $(unknown_ret_val)..."
    some_not_assigned_msg = "AbstractArray has some undefined references. Skipping over \
                             unassigned indices..."

    @eval begin
        function $(op)(x::AbstractArray{T}) where {T}
            if !isbitstype(T) && !(T <: Number)
                is_assigned_idxs = findall(Base.Fix1(isassigned, x), eachindex(x))
                if length(is_assigned_idxs) == 0
                    @warn $(all_not_assigned_msg)
                    return $(unknown_ret_val)
                elseif 0 < length(is_assigned_idxs) < length(x)
                    @warn $(some_not_assigned_msg)
                    x = x[is_assigned_idxs]
                end
                return mapreduce(MLDataDevices.$(op), combine_devices, x)
            end
            if hasmethod(parent, Tuple{typeof(x)})
                parent_x = parent(x)
                parent_x === x && return $(cpu_ret_val)
                return $(op)(parent_x)
            end
            return $(cpu_ret_val)
        end

        function $(op)(x::Union{Tuple,NamedTuple})
            length(x) == 0 && return $(op == :get_device ? nothing : Nothing)
            # NOTE: We need unrolled_mapreduce for julia 1.10 to ensure type stability
            return unrolled_mapreduce(MLDataDevices.$(op), combine_devices, values(x))
        end

        # NOTE: Don't mark as fast_structure
        $(op)(::Function) = $(op == :get_device ? UnknownDevice() : UnknownDevice)
    end

    for T in (Number, AbstractRNG, Val, Symbol, String, Nothing, AbstractRange)
        @eval $(op)(::$(T)) = $(op == :get_device ? nothing : Nothing)
    end
end

get_device(::Type{<:Number}) = CPUDevice()
get_device_type(::Type{<:Number}) = CPUDevice

get_device(_) = UnknownDevice()
get_device_type(_) = UnknownDevice

fast_structure(::AbstractArray) = true
fast_structure(::Union{Tuple,NamedTuple}) = true
for T in (Number, AbstractRNG, Val, Symbol, String, Nothing, AbstractRange)
    @eval fast_structure(::$(T)) = true
end
fast_structure(_) = false

function unrolled_mapreduce(f::F, op::O, itr) where {F,O}
    return unrolled_mapreduce(f, op, itr, static_length(itr))
end

function unrolled_mapreduce(::F, ::O, _, ::Val{0}) where {F,O}
    return error("Cannot unroll over an empty iterator.")
end

unrolled_mapreduce(f::F, ::O, itr, ::Val{1}) where {F,O} = f(only(itr))

@generated function unrolled_mapreduce(f::F, op::O, itr, ::Val{N}) where {F,O,N}
    syms = [gensym("f_itr_$(i)") for i in 1:N]
    op_syms = [gensym("op_$(i)") for i in 1:(N - 1)]
    f_applied = [:($(syms[i]) = f(itr[$i])) for i in 1:N]
    combine_expr = [:($(op_syms[1]) = op($(syms[1]), $(syms[2])))]
    for i in 2:(N - 1)
        push!(combine_expr, :($(op_syms[i]) = op($(op_syms[i - 1]), $(syms[i + 1]))))
    end
    return quote
        $(Expr(:meta, :inline))
        $(Expr(:inbounds, true))
        $(Expr(:block, f_applied...))
        $(Expr(:inbounds, :pop))
        $(Expr(:block, combine_expr...))
        return $(op_syms[end])
    end
end

function unsafe_free_internal!(x::AbstractArray)
    unsafe_free_internal!(MLDataDevices.get_device_type(x), x)
    return nothing
end
unsafe_free_internal!(::Type, x::AbstractArray) = nothing
unsafe_free_internal!(_) = nothing

function unsafe_free!(x)
    foreach(unsafe_free_internal!, fleaves(x))
    return nothing
end

static_length(t::Tuple) = Val(length(t))

function to_rarray(args...; kwargs...)
    loaded(ReactantDevice) && return to_rarray_internal(args...; kwargs...)
    return error("`to_rarray` is only supported with `Reactant` loaded.")
end

function to_rarray_internal end

# Utility function to facilitate data transfer
# For AbstractFloat and Complex{<:AbstractFloat} arrays, we provide specialized methods to avoid
# ambiguity with the general fallback and to enable efficient type conversion when needed.
function array_adapt(
    f::F, ::Type{aType}, ::Type{Missing}, x::AbstractArray{<:AbstractFloat}
) where {F,aType}
    return f(x)
end
function array_adapt(
    f::F, ::Type{aType}, ::Type{Missing}, x::AbstractArray{<:Complex{<:AbstractFloat}}
) where {F,aType}
    return f(x)
end

function array_adapt(
    ::F, ::Type{aType}, ::Type{Nothing}, x::AbstractArray{<:AbstractFloat}
) where {F,aType}
    return aType(x)
end
function array_adapt(
    ::F, ::Type{aType}, ::Type{Nothing}, x::AbstractArray{<:Complex{<:AbstractFloat}}
) where {F,aType}
    return aType(x)
end

# For specific type parameters, we do type conversion
function array_adapt(
    ::F, ::Type{aType}, ::Type{T}, x::AbstractArray{<:AbstractFloat}
) where {F,aType,T}
    return aType{T}(x)
end
function array_adapt(
    ::F, ::Type{aType}, ::Type{T}, x::AbstractArray{<:Complex{<:AbstractFloat}}
) where {F,aType,T}
    return aType{Complex{T}}(x)
end

# Fallback for all other isbits types (e.g., Int32, Char, or custom immutable structs)
function array_adapt(
    f::F, ::Type{aType}, ::Type{Missing}, x::AbstractArray{T}
) where {F,aType,T}
    isbitstype(T) || error(
        "Cannot move array with element type `$(T)` to device. Element type must be an \
        `isbits` type (e.g., Number, Char, or custom immutable struct with only \
        `isbits` fields)."
    )
    return f(x)
end

function array_adapt(
    ::F, ::Type{aType}, ::Type{Nothing}, x::AbstractArray{T}
) where {F,aType,T}
    isbitstype(T) || error(
        "Cannot move array with element type `$(T)` to device. Element type must be an \
        `isbits` type (e.g., Number, Char, or custom immutable struct with only \
        `isbits` fields)."
    )
    return aType(x)
end

function array_adapt(::F, ::Type{aType}, ::Type{E}, x::AbstractArray{T}) where {F,aType,E,T}
    isbitstype(T) || error(
        "Cannot move array with element type `$(T)` to device. Element type must be an \
        `isbits` type (e.g., Number, Char, or custom immutable struct with only \
        `isbits` fields)."
    )
    return aType(x)
end

end
