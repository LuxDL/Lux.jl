module DistributedUtils

using ChainRulesCore: ChainRulesCore
using Compat: @compat
using ConcreteStructs: @concrete
using Optimisers: Optimisers

using ..Lux: AbstractLuxDistributedBackend, MPIBackend, NCCLBackend
using MLDataDevices: get_device

const CRC = ChainRulesCore

const NCCL_Initialized = Ref(false)
const MPI_Initialized = Ref(false)

"""
    initialized(backend::Type{<:AbstractLuxDistributedBackend})

Check if the given backend is initialized.
"""
initialized(::Type{<:MPIBackend}) = MPI_Initialized[]
initialized(::Type{<:NCCLBackend}) = NCCL_Initialized[]

"""
    initialize(backend::Type{<:AbstractLuxDistributedBackend}; kwargs...)

Initialize the given backend. Users can supply `cuda_devices` and `amdgpu_devices` to
initialize the backend with the given devices. These can be set to `missing` to prevent
initialization of the given device type. If set to `nothing`, and the backend is functional
we assign GPUs in a round-robin fashion. Finally, a list of integers can be supplied to
initialize the backend with the given devices.

Possible values for `backend` are:

  - `MPIBackend`: MPI backend for distributed training. Requires `MPI.jl` to be installed.
  - `NCCLBackend`: NCCL backend for CUDA distributed training. Requires `CUDA.jl`,
    `MPI.jl`, and `NCCL.jl` to be installed. This also wraps `MPI` backend for non-CUDA
    communications.
"""
function initialize(backend::Type{<:AbstractLuxDistributedBackend}; kwargs...)
    initialized(backend) && return
    force_initialize(backend; kwargs...)
    return
end

function force_initialize end

"""
    get_distributed_backend(backend::Type{<:AbstractLuxDistributedBackend})

Get the distributed backend for the given backend type. Possible values are:

  - `MPIBackend`: MPI backend for distributed training. Requires `MPI.jl` to be installed.
  - `NCCLBackend`: NCCL backend for CUDA distributed training. Requires `CUDA.jl`,
    `MPI.jl`, and `NCCL.jl` to be installed. This also wraps `MPI` backend for non-CUDA
    communications.

!!! danger

    `initialize(backend; kwargs...)` must be called before calling this function.
"""
function get_distributed_backend(backend::Type{<:AbstractLuxDistributedBackend})
    initialized(backend) ||
        error("Backend `$(backend)` is not initialized. Call `DistributedUtils.initialize` first.")
    return unsafe_get_distributed_backend(backend)
end

function unsafe_get_distributed_backend end

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
    bcast_impl!(backend, sendrecvbuf, get_device(sendrecvbuf); root)
    return
end

function bcast!(backend::AbstractLuxDistributedBackend, sendbuf, recvbuf; root::Int=0)
    send_dev = get_device(sendbuf)
    recv_dev = get_device(recvbuf)
    if send_dev == recv_dev
        bcast_impl!(backend, sendbuf, recvbuf, send_dev; root)
    else
        sendbuf_ = sendbuf |> recv_dev
        @warn "`sendbuf` and `recvbuf` are on different devices." maxlog=1
        bcast_impl!(backend, sendbuf_, recvbuf, recv_dev; root)
    end
    return
end

function bcast_impl! end

CRC.@non_differentiable bcast!(::Any...)

function avg end

"""
    allreduce!(backend::AbstractLuxDistributedBackend, sendrecvbuf, op)
    allreduce!(backend::AbstractLuxDistributedBackend, sendbuf, recvbuf, op)

Backend Agnostic API to perform an allreduce operation on the given buffer `sendrecvbuf` or
`sendbuf` and store the result in `recvbuf`.

`op` allows a special `DistributedUtils.avg` operation that averages the result across all
workers.
"""
function allreduce!(backend::AbstractLuxDistributedBackend, sendrecvbuf, op::F) where {F}
    return allreduce_impl!(backend, sendrecvbuf, op, get_device(sendrecvbuf))
end

function allreduce!(
        backend::AbstractLuxDistributedBackend, sendbuf, recvbuf, op::F) where {F}
    send_dev = get_device(sendbuf)
    recv_dev = get_device(recvbuf)
    if send_dev == recv_dev
        allreduce_impl!(backend, sendbuf, recvbuf, op, send_dev)
    else
        sendbuf_ = sendbuf |> recv_dev
        @warn "`sendbuf` and `recvbuf` are on different devices." maxlog=1
        allreduce_impl!(backend, sendbuf_, recvbuf, op, recv_dev)
    end
    return
end

function allreduce_impl! end

CRC.@non_differentiable allreduce!(::Any...)

"""
    reduce!(backend::AbstractLuxDistributedBackend, sendrecvbuf, op; root::Int=0)
    reduce!(backend::AbstractLuxDistributedBackend, sendbuf, recvbuf, op; root::Int=0)

Backend Agnostic API to perform a reduce operation on the given buffer `sendrecvbuf` or
`sendbuf` and store the result in `recvbuf`.

`op` allows a special `DistributedUtils.avg` operation that averages the result across all
workers.
"""
function reduce!(
        backend::AbstractLuxDistributedBackend, sendrecvbuf, op::F; root::Int=0) where {F}
    return reduce_impl!(backend, sendrecvbuf, op, get_device(sendrecvbuf); root)
end

function reduce!(backend::AbstractLuxDistributedBackend,
        sendbuf, recvbuf, op::F; root::Int=0) where {F}
    send_dev = get_device(sendbuf)
    recv_dev = get_device(recvbuf)
    if send_dev == recv_dev
        reduce_impl!(backend, sendbuf, recvbuf, op, send_dev; root)
    else
        sendbuf_ = sendbuf |> recv_dev
        @warn "`sendbuf` and `recvbuf` are on different devices." maxlog=1
        reduce_impl!(backend, sendbuf_, recvbuf, op, recv_dev; root)
    end
    return
end

function reduce_impl! end

CRC.@non_differentiable reduce!(::Any...)

# synchronize!
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

function synchronize!!(backend::AbstractLuxDistributedBackend, ps::T; root::Int=0) where {T}
    if isbitstype(T) || T <: Number
        psₙ = [ps]
        bcast!(backend, psₙ; root)
        return psₙ[]
    end
    return ps # If we don't know how to synchronize, just return the value. For ex, Symbol, String, etc.
end

# data container
"""
    DistributedDataContainer(backend::AbstractLuxDistributedBackend, data)

`data` must be compatible with `MLUtils` interface. The returned container is compatible
with `MLUtils` interface and is used to partition the dataset across the available
processes.

!!! danger "Load `MLUtils.jl`"

    `MLUtils.jl` must be installed and loaded before using this.
"""
@concrete struct DistributedDataContainer
    data
    idxs
end

function DistributedDataContainer(backend::AbstractLuxDistributedBackend, data)
    if Base.get_extension(@__MODULE__, :LuxMLUtilsExt) === nothing
        error("`MLUtils.jl` must be installed and loaded before using \
               `DistributedDataContainer`.")
    end
    return construct_distributed_data_container(backend, data)
end

function construct_distributed_data_container end

Base.length(ddc::DistributedDataContainer) = length(ddc.idxs)

Base.getindex(ddc::DistributedDataContainer, i) = getindex(ddc.data, ddc.idxs[i])

# Distributed Optimizer
"""
    DistributedOptimizer(backend::AbstractLuxDistributedBacked, optimizer)

Wrap the `optimizer` in a `DistributedOptimizer`. Before updating the parameters, this
averages the gradients across the processes using Allreduce.

## Arguments

  - `optimizer`: An Optimizer compatible with the Optimisers.jl package
"""
@concrete struct DistributedOptimizer <: Optimisers.AbstractRule
    backend <: AbstractLuxDistributedBackend
    opt <: Optimisers.AbstractRule
end

function Optimisers.apply!(opt::DistributedOptimizer, state, x, y)
    allreduce!(opt.backend, y, avg)
    return Optimisers.apply!(opt.opt, state, x, y)
end

Optimisers.init(opt::DistributedOptimizer, x::AbstractArray) = Optimisers.init(opt.opt, x)

function Optimisers._adjust(opt::DistributedOptimizer, nt::NamedTuple)
    return DistributedOptimizer(opt.backend, Optimisers._adjust(opt.opt, nt))
end

function synchronize!!(
        backend::AbstractLuxDistributedBackend, ps::Optimisers.Leaf; root::Int=0)
    return Optimisers.Leaf(ps.rule, synchronize!!(backend, ps.state; root), ps.frozen)
end

@compat(public,
    (initialized, initialize, get_distributed_backend, local_rank,
    total_workers, bcast!, allreduce!, reduce!, synchronize!!,
    DistributedDataContainer, DistributedOptimizer, avg))

end
