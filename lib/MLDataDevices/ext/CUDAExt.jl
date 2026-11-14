module CUDAExt

using Adapt: Adapt
using CUDA: CUDA, CuArray
using MLDataDevices: MLDataDevices, Internal, CUDADevice, CPUDevice
using Random: Random

Internal.with_device(::Type{CUDADevice}, ::Nothing) = CUDADevice(nothing)
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
Internal.get_device(::Type{<:CUDA.AnyCuArray}) = CUDADevice(CUDA.device())
Internal.get_device(::CUDA.RNG) = CUDADevice(CUDA.device())

Internal.get_device_type(::CUDA.AnyCuArray) = CUDADevice
Internal.get_device_type(::Type{<:CUDA.AnyCuArray}) = CUDADevice
Internal.get_device_type(::CUDA.RNG) = CUDADevice

const CURAND_RNG = Union{CUDA.cuRAND.LibraryRNG,CUDA.cuRAND.NativeRNG}
Internal.get_device_type(::CURAND_RNG) = CUDADevice
Internal.get_device(::CURAND_RNG) = CUDADevice(CUDA.device())

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
function with_cuda_device(f::F, target_dev::CUDA.CuDevice) where {F}
    old_dev = CUDA.device()
    old_dev == target_dev && return f()
    CUDA.device!(target_dev)
    try
        return f()
    finally
        CUDA.device!(old_dev)
    end
end
with_cuda_device(f::F, ::Nothing) where {F} = f()

const STAGING_TLS_KEY = :_MLDataDevices_CUDA_Staging
const MAX_STAGING_SLOTS = 8

mutable struct CUDAStagingSlot
    host_buf::Any
    copy_ev::Union{Nothing,CUDA.CuEvent}
end

mutable struct CUDAStagingPool
    const slots::Vector{CUDAStagingSlot}
    next_evict::Int
end

function is_bfloat16(::Type{T}) where {T}
    @static if isdefined(Core, :BFloat16)
        T === Core.BFloat16 && return true
    end
    return T <: AbstractFloat && sizeof(T) == 2 && nameof(T) === :BFloat16
end

cuda_target_eltype(::Type{Nothing}, ::Type{T}) where {T} = T

function cuda_target_eltype(::Type{Missing}, ::Type{T}) where {T}
    T === Float16 && return Float16
    is_bfloat16(T) && return T
    T <: AbstractFloat && return Float32
    T <: Complex{<:AbstractFloat} && return ComplexF32
    return T
end

function cuda_target_eltype(::Type{E}, ::Type{T}) where {E,T}
    T <: AbstractFloat && return E
    T <: Complex{<:AbstractFloat} && return Complex{E}
    return T
end

_is_stageable_host_array(::Array) = true
_is_stageable_host_array(::BitArray) = true
_is_stageable_host_array(::AbstractArray) = false

function acquire_host_buffer!(::Type{T}, len::Int) where {T}
    device_pools = get!(task_local_storage(), STAGING_TLS_KEY) do
        return Dict{Cint,CUDAStagingPool}()
    end
    dev_id = Cint(CUDA.deviceid(CUDA.device()))
    pool = get!(device_pools, dev_id) do
        return CUDAStagingPool(CUDAStagingSlot[], 1)
    end
    slots = pool.slots

    fit_slot = nothing
    same_type_slot = nothing
    for slot in slots
        if slot.copy_ev === nothing || CUDA.isdone(slot.copy_ev)
            if slot.host_buf isa CuArray{T,1,CUDA.HostMemory}
                if length(slot.host_buf) >= len
                    fit_slot = slot
                    break
                elseif same_type_slot === nothing
                    same_type_slot = slot
                end
            end
        end
    end

    if fit_slot !== nothing
        chosen = fit_slot
    elseif same_type_slot !== nothing
        chosen = same_type_slot
    elseif length(slots) < MAX_STAGING_SLOTS
        buf = CuArray{T,1,CUDA.HostMemory}(undef, len)
        ev = CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING)
        slot = CUDAStagingSlot(buf, ev)
        push!(slots, slot)
        return slot
    else
        chosen = nothing
        for slot in slots
            if slot.copy_ev === nothing || CUDA.isdone(slot.copy_ev)
                chosen = slot
                break
            end
        end
        if chosen === nothing
            chosen = slots[pool.next_evict]
            pool.next_evict = mod1(pool.next_evict + 1, length(slots))
            chosen.copy_ev !== nothing &&
                (CUDA.isdone(chosen.copy_ev) || CUDA.synchronize(chosen.copy_ev))
        end
    end

    if !(chosen.host_buf isa CuArray{T,1,CUDA.HostMemory} && length(chosen.host_buf) >= len)
        if chosen.host_buf !== nothing
            old_buf = chosen.host_buf
            chosen.host_buf = nothing
            CUDA.unsafe_free!(old_buf)
        end
        chosen.host_buf = CuArray{T,1,CUDA.HostMemory}(undef, len)
    end
    if chosen.copy_ev === nothing
        chosen.copy_ev = CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING)
    end
    return chosen
end

function stage_and_upload!(::Type{T_dst}, x::AbstractArray) where {T_dst}
    len = length(x)
    slot = acquire_host_buffer!(T_dst, len)
    gpu_dest = CuArray{T_dst}(undef, size(x))
    buf = slot.host_buf::CuArray{T_dst,1,CUDA.HostMemory}
    GC.@preserve buf begin
        host_ptr = pointer(buf; type=CUDA.HostMemory)
        host_arr = unsafe_wrap(Array, host_ptr, size(x))
        copyto!(host_arr, x)
        copyto!(gpu_dest, 1, buf, 1, len)
    end
    CUDA.record(slot.copy_ev, CUDA.stream())
    return gpu_dest
end

function cuda_array_adapt(::CUDADevice{D,E}, x::AbstractArray{T}) where {D,E,T}
    if isbitstype(T) && !isempty(x) && _is_stageable_host_array(x)
        T_dst = cuda_target_eltype(E, T)
        if sizeof(T_dst) > 0
            return stage_and_upload!(T_dst, x)
        end
    end
    return Internal.array_adapt(CUDA.cu, CuArray, E, x)
end

function Adapt.adapt_storage(to::CUDADevice{D,E}, x::AbstractArray) where {D,E}
    dev = MLDataDevices.get_device(x)
    if !(dev isa CUDADevice)
        return with_cuda_device(to.device) do
            return cuda_array_adapt(to, x)
        end
    elseif dev.device === nothing || to.device === nothing || dev.device == to.device
        Internal.return_without_conversion(E, x) && return x
        return with_cuda_device(to.device) do
            return cuda_array_adapt(to, x)
        end
    else
        return with_cuda_device(to.device) do
            if Internal.return_without_conversion(E, x)
                return copy(x)
            else
                return cuda_array_adapt(to, copy(x))
            end
        end
    end
end

# Prefetching Backend for DeviceIterator
mutable struct CUDAPrefetchBackend
    const device::CUDA.CuDevice
    const stream::CUDA.CuStream
    const lock::ReentrantLock
    const free_events::Vector{CUDA.CuEvent}
    const retired_events::Vector{CUDA.CuEvent}
    const staging_pools::Dict{Cint,CUDAStagingPool}
end

Internal.supports_prefetch(::CUDADevice) = CUDA.functional()

function Internal.init_prefetch_backend(dev::CUDADevice, prefetch::Int)
    CUDA.functional() || return nothing
    target_dev = dev.device === nothing ? CUDA.device() : dev.device
    stream = with_cuda_device(target_dev) do
        return CUDA.CuStream(; flags=CUDA.STREAM_NON_BLOCKING)
    end
    return CUDAPrefetchBackend(
        target_dev,
        stream,
        ReentrantLock(),
        CUDA.CuEvent[],
        CUDA.CuEvent[],
        Dict{Cint,CUDAStagingPool}(),
    )
end

function acquire_event!(backend::CUDAPrefetchBackend)
    ev = Base.@lock backend.lock begin
        isempty(backend.free_events) ? nothing : pop!(backend.free_events)
    end
    return ev === nothing ? CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING) : ev
end

function recycle_event!(backend::CUDAPrefetchBackend, ev::CUDA.CuEvent)
    Base.@lock backend.lock begin
        push!(backend.free_events, ev)
    end
    return nothing
end

function drain_retired_events!(backend::CUDAPrefetchBackend)
    while true
        ev = Base.@lock backend.lock begin
            length(backend.retired_events) > 1 ? popfirst!(backend.retired_events) : nothing
        end
        ev === nothing && break
        CUDA.isdone(ev) || CUDA.synchronize(ev)
        recycle_event!(backend, ev)
    end
    return nothing
end

function Internal.with_prefetch_context(f::F, backend::CUDAPrefetchBackend) where {F}
    return with_cuda_device(backend.device) do
        return CUDA.stream!(backend.stream) do
            tls = task_local_storage()
            has_prev = haskey(tls, STAGING_TLS_KEY)
            prev_pools = has_prev ? tls[STAGING_TLS_KEY] : nothing
            tls[STAGING_TLS_KEY] = backend.staging_pools
            try
                return f()
            finally
                if has_prev
                    tls[STAGING_TLS_KEY] = prev_pools
                else
                    delete!(tls, STAGING_TLS_KEY)
                end
            end
        end
    end
end

function Internal.prefetch_transfer!(dev::CUDADevice, backend::CUDAPrefetchBackend, batch)
    drain_retired_events!(backend)
    dev_batch = dev(batch)
    ready_ev = acquire_event!(backend)
    CUDA.record(ready_ev, backend.stream)
    return dev_batch, ready_ev
end

function Internal.prefetch_consume!(backend::CUDAPrefetchBackend, ready_ev::CUDA.CuEvent)
    CUDA.isdone(ready_ev) || CUDA.synchronize(ready_ev)
    CUDA.wait(ready_ev, CUDA.stream())
    recycle_event!(backend, ready_ev)
    return nothing
end

function Internal.prefetch_retire!(backend::CUDAPrefetchBackend, prev_batch)
    Internal.unsafe_free!(prev_batch)
    if CUDA.device() == backend.device
        ev = acquire_event!(backend)
        CUDA.record(ev, CUDA.stream())
        Base.@lock backend.lock begin
            push!(backend.retired_events, ev)
        end
    end
    return nothing
end

function Internal.cleanup_prefetch_backend!(backend::CUDAPrefetchBackend)
    for pool in values(backend.staging_pools)
        for slot in pool.slots
            if slot.copy_ev !== nothing
                CUDA.isdone(slot.copy_ev) || CUDA.synchronize(slot.copy_ev)
            end
            if slot.host_buf !== nothing
                old_buf = slot.host_buf
                slot.host_buf = nothing
                CUDA.unsafe_free!(old_buf)
            end
        end
        empty!(pool.slots)
    end
    empty!(backend.staging_pools)
    return nothing
end

Adapt.adapt_storage(::CPUDevice, rng::CUDA.RNG) = Random.default_rng()

end
