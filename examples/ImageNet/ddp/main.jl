using Zygote

using LuxCUDA # NVIDIA GPUs
# using AMDGPU # Install and load AMDGPU to train models on AMD GPUs with ROCm

using MPI: MPI
using NCCL: NCCL # Enables distributed training in Lux. NCCL is needed for CUDA GPUs

include(joinpath(@__DIR__, "../common.jl"))

is_distributed() = false
should_log() = true

const gdev = gpu_device()
const cdev = cpu_device()

# Distributed Training: NCCL for NVIDIA GPUs and MPI for anything else
const distributed_backend = try
    if gdev isa CUDADevice
        DistributedUtils.initialize(NCCLBackend)
        DistributedUtils.get_distributed_backend(NCCLBackend)
    else
        DistributedUtils.initialize(MPIBackend)
        DistributedUtils.get_distributed_backend(MPIBackend)
    end
catch err
    path_to_single_host = joinpath(@__DIR__, "../single_host/main.jl")
    @error "Could not initialize distributed training. If you want to use Single Host \
            training/inference, use $(path_to_single_host) instead."
    rethrow()
end

const local_rank = DistributedUtils.local_rank(distributed_backend)
const total_workers = DistributedUtils.total_workers(distributed_backend)

@assert total_workers > 1 "There should be more than one worker for DDP training."

is_distributed() = true
should_log() = local_rank == 0

# Change our logger to only log from rank 0
const logger = FormatLogger() do io, args
    if should_log()
        println(
            io,
            args._module,
            " | ",
            Dates.format(now(), "yyyy-mm-dd HH:MM:SS"),
            " | [",
            args.level,
            "] ",
            args.message,
        )
    end
end
global_logger(logger)
