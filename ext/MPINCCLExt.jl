module MPINCCLExt

using MPI: MPI
using NCCL: NCCL
using Setfield: @set!

using Lux: Lux, MPIBackend, NCCLBackend, DistributedUtils
using MLDataDevices: AbstractDevice, CUDADevice

Lux.is_extension_loaded(::Val{:MPINCCL}) = true

function DistributedUtils.force_initialize(
    ::Type{NCCLBackend}; cuda_devices=nothing, amdgpu_devices=missing
)
    @assert amdgpu_devices === missing "`AMDGPU` is not supported by `NCCL`."
    DistributedUtils.force_initialize(
        MPIBackend; cuda_devices, force_cuda=true, caller="NCCLBackend", amdgpu_devices
    )
    DistributedUtils.NCCL_Initialized[] = true
    return nothing
end

function DistributedUtils.unsafe_get_distributed_backend(::Type{NCCLBackend})
    unique_id = NCCL.UniqueID()  # Generate on all ranks to know the type
    mpi_backend = DistributedUtils.unsafe_get_distributed_backend(MPIBackend)
    buf = [unique_id.internal...]
    DistributedUtils.bcast!(mpi_backend, buf; root=0)
    @set! unique_id.internal = Tuple(buf)

    nranks = DistributedUtils.total_workers(mpi_backend)
    rank = DistributedUtils.local_rank(mpi_backend)

    return NCCLBackend(NCCL.Communicator(nranks, rank; unique_id), mpi_backend)
end

DistributedUtils.local_rank(backend::NCCLBackend) = NCCL.rank(backend.comm)

DistributedUtils.total_workers(backend::NCCLBackend) = NCCL.size(backend.comm)

# For non-CUDA Arrays, fallback to MPI
# Broadcast
function DistributedUtils.bcast_impl!(
    backend::NCCLBackend, sendrecvbuf, ::CUDADevice; root=0
)
    NCCL.Broadcast!(sendrecvbuf, backend.comm; root)
    return nothing
end

function DistributedUtils.bcast_impl!(
    backend::NCCLBackend, sendrecvbuf, dev::AbstractDevice; root=0
)
    DistributedUtils.bcast_impl!(backend.mpi_backend, sendrecvbuf, dev; root)
    return nothing
end

function DistributedUtils.bcast_impl!(
    backend::NCCLBackend, sendbuf, recvbuf, ::CUDADevice; root=0
)
    NCCL.Broadcast!(sendbuf, recvbuf, backend.comm; root)
    return nothing
end

function DistributedUtils.bcast_impl!(
    backend::NCCLBackend, sendbuf, recvbuf, dev::AbstractDevice; root=0
)
    DistributedUtils.bcast_impl!(backend.mpi_backend, sendbuf, recvbuf, dev; root)
    return nothing
end

# Allreduce
function DistributedUtils.allreduce_impl!(
    backend::NCCLBackend, sendrecvbuf, op::F, ::CUDADevice
) where {F}
    op = ifelse(op === DistributedUtils.avg, NCCL.avg, op)
    NCCL.Allreduce!(sendrecvbuf, op, backend.comm)
    return nothing
end

function DistributedUtils.allreduce_impl!(
    backend::NCCLBackend, sendrecvbuf, op::F, dev::AbstractDevice
) where {F}
    DistributedUtils.allreduce_impl!(backend.mpi_backend, sendrecvbuf, op, dev)
    return nothing
end

function DistributedUtils.allreduce_impl!(
    backend::NCCLBackend, sendbuf, recvbuf, op::F, ::CUDADevice
) where {F}
    op = ifelse(op === DistributedUtils.avg, NCCL.avg, op)
    NCCL.Allreduce!(sendbuf, recvbuf, op, backend.comm)
    return nothing
end

function DistributedUtils.allreduce_impl!(
    backend::NCCLBackend, sendbuf, recvbuf, op::F, dev::AbstractDevice
) where {F}
    DistributedUtils.allreduce_impl!(backend.mpi_backend, sendbuf, recvbuf, op, dev)
    return nothing
end

# Reduce
function DistributedUtils.reduce_impl!(
    backend::NCCLBackend, sendrecvbuf, op::F, ::CUDADevice; root::Int
) where {F}
    op = ifelse(op === DistributedUtils.avg, NCCL.avg, op)
    NCCL.Reduce!(sendrecvbuf, op, backend.comm; root)
    return nothing
end

function DistributedUtils.reduce_impl!(
    backend::NCCLBackend, sendrecvbuf, op::F, dev::AbstractDevice; root::Int
) where {F}
    DistributedUtils.reduce_impl!(backend.mpi_backend, sendrecvbuf, op, dev; root)
    return nothing
end

function DistributedUtils.reduce_impl!(
    backend::NCCLBackend, sendbuf, recvbuf, op::F, ::CUDADevice; root::Int
) where {F}
    op = ifelse(op === DistributedUtils.avg, NCCL.avg, op)
    NCCL.Reduce!(sendbuf, recvbuf, op, backend.comm; root)
    return nothing
end

function DistributedUtils.reduce_impl!(
    backend::NCCLBackend, sendbuf, recvbuf, op::F, dev::AbstractDevice; root::Int
) where {F}
    DistributedUtils.reduce_impl!(backend.mpi_backend, sendbuf, recvbuf, op, dev; root)
    return nothing
end

end
