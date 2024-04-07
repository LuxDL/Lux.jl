using Lux, MPI, NCCL, Test

const input_args = length(ARGS) == 2 ? ARGS : ("CPU", "mpi")
const backend_type = input_args[2] == "nccl" ? NCCLBackend : MPIBackend
const dev = input_args[1] == "CPU" ? LuxCPUDevice() :
            (input_args[1] == "CUDA" ? LuxCUDADevice() : LuxAMDGPUDevice())

DistributedUtils.initialize(backend_type)
backend = DistributedUtils.get_distributed_backend(backend_type)

@test DistributedUtils.initialized(backend_type)

# Should always hold true
@test DistributedUtils.local_rank(backend) < DistributedUtils.total_workers(backend)

# Test the communication primitives