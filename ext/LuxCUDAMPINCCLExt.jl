module LuxCUDAMPINCCLExt

import Lux: MPIBackend, NCCLBackend, DistributedUtils
import MPI
import NCCL
import Setfield: @set!

function DistributedUtils.__initialize(
        ::Val{:NCCL}; cuda_devices=nothing, amdgpu_devices=missing)
    DistributedUtils.NCCL_Initialized[] = true
    @assert amdgpu_devices === missing "`AMDGPU` is not supported by `NCCL`."
    DistributedUtils.__initialize(Val(:MPI); cuda_devices, amdgpu_devices)
    return
end

function DistributedUtils.get_distributed_backend(::Val{:NCCL})
    unique_id = NCCL.UniqueID()  # Generate on all ranks to know the type
    mpi_backend = DistributedUtils.get_distributed_backend(Val(:MPI))
    buf = [unique_id.internal...]
    DistributedUtils.bcast!(mpi_backend, buf; root=0)
    @set! unique_id.internal = Tuple(buf)

    nranks = DistributedUtils.total_workers(mpi_backend)
    rank = DistributedUtils.local_rank(mpi_backend)

    return NCCLBackend(NCCL.Communicator(nranks, rank; unique_id))
end

DistributedUtils.local_rank(backend::NCCLBackend) = NCCL.rank(backend.comm)

DistributedUtils.total_workers(backend::NCCLBackend) = NCCL.size(backend.comm)

function DistributedUtils.bcast!(backend::NCCLBackend, sendrecvbuf; root=0)
    NCCL.Broadcast!(sendrecvbuf, backend.comm; root)
    return sendrecvbuf
end

function DistributedUtils.bcast!(backend::NCCLBackend, sendbuf, recvbuf; root=0)
    NCCL.Broadcast!(sendbuf, recvbuf, backend.comm; root)
    return recvbuf
end

end
