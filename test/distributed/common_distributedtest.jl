using Lux, MPI, Test

const input_args = length(ARGS) == 2 ? ARGS : ("cpu", "mpi")

if input_args[1] == "cuda"
    using LuxCUDA
end
if input_args[1] == "amdgpu"
    using AMDGPU
end

const backend_type = input_args[2] == "nccl" ? NCCLBackend : MPIBackend

if input_args[2] == "nccl"
    using NCCL
end

const dev = if input_args[1] == "cpu"
    CPUDevice()
else
    (input_args[1] == "cuda" ? CUDADevice() : AMDGPUDevice())
end
const aType =
    input_args[1] == "cpu" ? Array : (input_args[1] == "cuda" ? CuArray : ROCArray)

DistributedUtils.initialize(backend_type)
backend = DistributedUtils.get_distributed_backend(backend_type)

@test DistributedUtils.initialized(backend_type)

# Should always hold true
rank = DistributedUtils.local_rank(backend)
nworkers = DistributedUtils.total_workers(backend)
@test rank < nworkers

# Test the communication primitives
## broadcast!
for arrType in (Array, aType)
    sendbuf = (rank == 0) ? arrType(ones(512)) : arrType(zeros(512))
    recvbuf = arrType(zeros(512))

    DistributedUtils.bcast!(backend, sendbuf, recvbuf; root=0)

    rank != 0 && @test all(recvbuf .== 1)

    sendrecvbuf = (rank == 0) ? arrType(ones(512)) : arrType(zeros(512))
    DistributedUtils.bcast!(backend, sendrecvbuf; root=0)

    @test all(sendrecvbuf .== 1)
end

## reduce!
for arrType in (Array, aType)
    sendbuf = arrType(fill(Float64(rank + 1), 512))
    recvbuf = arrType(zeros(512))

    DistributedUtils.reduce!(backend, sendbuf, recvbuf, +; root=0)

    rank == 0 && @test all(recvbuf .≈ sum(1:nworkers))

    sendbuf .= rank + 1

    DistributedUtils.reduce!(backend, sendbuf, recvbuf, DistributedUtils.avg; root=0)

    rank == 0 && @test all(recvbuf .≈ sum(1:nworkers) / nworkers)

    sendrecvbuf = arrType(fill(Float64(rank + 1), 512))

    DistributedUtils.reduce!(backend, sendrecvbuf, +; root=0)

    rank == 0 && @test all(sendrecvbuf .≈ sum(1:nworkers))

    sendrecvbuf .= rank + 1

    DistributedUtils.reduce!(backend, sendrecvbuf, DistributedUtils.avg; root=0)

    rank == 0 && @test all(sendrecvbuf .≈ sum(1:nworkers) / nworkers)
end

## allreduce!
for arrType in (Array, aType)
    sendbuf = arrType(fill(Float64(rank + 1), 512))
    recvbuf = arrType(zeros(512))

    DistributedUtils.allreduce!(backend, sendbuf, recvbuf, +)

    @test all(recvbuf .≈ sum(1:nworkers))

    sendbuf .= rank + 1

    DistributedUtils.allreduce!(backend, sendbuf, recvbuf, DistributedUtils.avg)

    @test all(recvbuf .≈ sum(1:nworkers) / nworkers)

    sendrecvbuf = arrType(fill(Float64(rank + 1), 512))

    DistributedUtils.allreduce!(backend, sendrecvbuf, +)

    @test all(sendrecvbuf .≈ sum(1:nworkers))

    sendrecvbuf .= rank + 1

    DistributedUtils.allreduce!(backend, sendrecvbuf, DistributedUtils.avg)

    @test all(sendrecvbuf .≈ sum(1:nworkers) / nworkers)
end
