module DistributedUtils

import ChainRulesCore as CRC
import Functors: fmap
import ..Lux: AbstractLuxDistributedBackend, MPIBackend, NCCLBackend
import LuxDeviceUtils: get_device, cpu_device
import Optimisers: Leaf
import Setfield: @set!

const NCCL_Initialized = Ref(false)
const MPI_Initialized = Ref(false)

"""
    initialized(backend::Val)

Check if the given backend is initialized.
"""
initialized(::Val{:MPI}) = MPI_Initialized[]
initialized(::Val{:NCCL}) = NCCL_Initialized[]

"""
    initialize(backend::Val; kwargs...)

Initialize the given backend. Users can supply `cuda_devices` and `amdgpu_devices` to
initialize the backend with the given devices. These can be set to `missing` to prevent
initialization of the given device type. If set to `nothing`, and the backend is functional
we assign GPUs in a round-robin fashion. Finally, a list of integers can be supplied to
initialize the backend with the given devices.

Possible values for `backend` are:

  - `Val(:MPI)`: MPI backend for distributed training. Requires `MPI.jl` to be installed.
  - `Val(:NCCL)`: NCCL backend for CUDA distributed training. Requires `CUDA.jl`,
    `MPI.jl`, and `NCCL.jl` to be installed. This also wraps `MPI` backend for non-CUDA
    communications.
"""
function initialize(backend::Val; kwargs...)
    initialized(backend) && return
    __initialize(backend; kwargs...)
    return
end

function __initialize end

"""
    get_distributed_backend(backend::Val)

Get the distributed backend for the given backend type. Possible values are:

  - `Val(:MPI)`: MPI backend for distributed training. Requires `MPI.jl` to be installed.
  - `Val(:NCCL)`: NCCL backend for CUDA distributed training. Requires `CUDA.jl`,
    `MPI.jl`, and `NCCL.jl` to be installed. This also wraps `MPI` backend for non-CUDA
    communications.

!!! danger

    `initialize(backend; kwargs...)` must be called before calling this function.
"""
function get_distributed_backend(backend::Val)
    initialized(backend) ||
        error("Backend `$(backend)` is not initialized. Call `DistributedUtils.initialize` first.")
    return __get_distributed_backend(backend)
end

function __get_distributed_backend end

CRC.@non_differentiable get_distributed_backend(::Any...)

"""
    local_rank(backend::AbstractLuxDistributedBackend)

Get the local rank for the given backend.
"""
function local_rank end

CRC.@non_differentiable local_rank(::Any...)

"""
    total_workers(backend::AbstractLuxDistributedBackend)

Get the total number of workers for the given backend.
"""
function total_workers end

CRC.@non_differentiable total_workers(::Any...)

"""
    bcast!(backend::AbstractLuxDistributedBackend, sendrecvbuf; root::Int=0)
    bcast!(backend::AbstractLuxDistributedBackend, sendbuf, recvbuf; root::Int=0)

Backend Agnostic API to broadcast the given buffer `sendrecvbuf` or `sendbuf` to all
workers into `recvbuf`. The value at `root` will be broadcasted to all other workers.
"""
function bcast!(backend::AbstractLuxDistributedBackend, sendrecvbuf; root::Int=0)
    return __bcast!(backend, sendrecvbuf, get_device(sendrecvbuf); root)
end

function bcast!(backend::AbstractLuxDistributedBackend, sendbuf, recvbuf; root::Int=0)
    send_dev = get_device(sendbuf)
    recv_dev = get_device(recvbuf)
    if send_dev === recv_dev
        return __bcast!(backend, sendbuf, recvbuf, send_dev; root)
    else
        cdev = cpu_device()
        sendbuf_ = sendbuf |> cdev
        recvbuf_ = recvbuf |> cdev
        @warn "`sendbuf` and `recvbuf` are on different devices. Moving them to `CPU`." maxlog=1
        __bcast!(backend, sendbuf_, recvbuf_, cdev; root)
        return recvbuf_ |> recv_dev
    end
end

function __bcast! end

CRC.@non_differentiable bcast!(::Any...)

function allreduce! end

function __allreduce! end

CRC.@non_differentiable allreduce!(::Any...)

function reduce! end

function __reduce! end

CRC.@non_differentiable reduce!(::Any...)

# syncronize!
"""
    synchronize!!(backend::AbstractLuxDistributedBackend, ps; root::Int=0)

Synchronize the given structure `ps` using the given backend. The value at `root` will be
broadcasted to all other workers.
"""
function synchronize!!(backend::AbstractLuxDistributedBackend, ps::Tuple; root::Int=0)
    length(ps) == 0 && return ps
    return map(x -> synchronize!!(backend, x; root), ps)
end

function synchronize!!(backend::AbstractLuxDistributedBackend,
        ps::NamedTuple{fields}; root::Int=0) where {fields}
    length(ps) == 0 && return ps
    return NamedTuple{fields}(map(x -> synchronize!!(backend, x; root), values(ps)))
end

function synchronize!!(
        backend::AbstractLuxDistributedBackend, ps::AbstractArray{T}; root::Int=0) where {T}
    if isbitstype(T)
        bcast!(backend, ps; root)
        return ps
    end
    return map(x -> synchronize!!(backend, x; root), ps)
end

function synchronize!!(backend::AbstractLuxDistributedBackend, ps::Leaf; root::Int=0)
    @set! ps.state = synchronize!!(backend, ps.state; root)
    return ps
end

function synchronize!!(backend::AbstractLuxDistributedBackend, ps::T; root::Int=0) where {T}
    isbitstype(T) && return bcast!(backend, [ps]; root)[]
    return ps # If we don't know how to synchronize, just return the value. For ex, Symbol, String, etc.
end

# data container
"""
    DistributedDataContainer(backend::AbstractLuxDistributedBackend, data)

`data` must be compatible with `MLUtils` interface. The returned container is compatible
with `MLUtils` interface and is used to partition the dataset across the available
processes.
"""
struct DistributedDataContainer{D, I}
    data::D
    indxs::I
end

function DistributedDataContainer(backend::AbstractLuxDistributedBackend, data)
    total_size = length(data)
    split_across = total_length(backend)
    size_per_worker = Int(ceil(total_size / split_across))

    partitions = collect(Iterators.partition(1:total_size, size_per_worker))
    idxs = collect(partitions[local_rank(backend) + 1])

    return DistributedDataContainer(data, idxs)
end

Base.length(ddc::DistributedDataContainer) = length(ddc.idxs)

Base.getindex(ddc::DistributedDataContainer, i) = getindex(ddc.data, ddc.idxs[i])

end
