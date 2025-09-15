using Lux, MPI, Random, Test

@test_throws ErrorException DistributedUtils.DistributedDataContainer(
    MPIBackend(), randn(Float32, 10)
)

using MLUtils

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

rng = Xoshiro(1234)

DistributedUtils.initialize(backend_type)
backend = DistributedUtils.get_distributed_backend(backend_type)

data = randn(rng, Float32, 10)
dcontainer = DistributedUtils.DistributedDataContainer(backend, data)

@test dcontainer[1] isa Any

rank = DistributedUtils.local_rank(backend)
tworkers = DistributedUtils.total_workers(backend)

if rank != tworkers - 1
    @test length(dcontainer) == ceil(length(data) / tworkers)
else
    @test length(dcontainer) ==
        length(data) - (tworkers - 1) * ceil(length(data) / tworkers)
end

dsum = sum(Base.Fix1(MLUtils.getobs, dcontainer), 1:MLUtils.numobs(dcontainer))
dsum_arr = [dsum]
DistributedUtils.allreduce!(backend, dsum_arr, +)
@test dsum_arr[1] ≈ sum(data)
