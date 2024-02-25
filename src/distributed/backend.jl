abstract type AbstractLuxDistributedBackend end

"""
    MPIBackend(comm = nothing)

Create an MPI backend for distributed training. Users should not use this function directly.
Instead use [`DistributedUtils.get_distributed_backend(Val(:NCCL))`](@ref).
"""
struct MPIBackend{C} <: AbstractLuxDistributedBackend
    comm::C

    function MPIBackend(comm=nothing)
        if Base.get_extension(@__MODULE__, :LuxMPIExt) === nothing
            error("`MPIBackend` requires `MPI.jl` to be loaded.")
        end
        return new{typeof(comm)}(comm)
    end
end

struct NCCLBackend{C, M <: Union{Nothing, MPIBackend}} <: AbstractLuxDistributedBackend
    comm::C
    mpi_backend::M

    function NCCLBackend(comm=nothing, mpi_backend=nothing)
        if Base.get_extension(@__MODULE__, :LuxCUDAMPINCCLExt) === nothing
            error("`NCCLBackend` requires `CUDA.jl`, `MPI.jl` and `NCCL.jl` to be loaded.")
        end
        return new{typeof(comm), typeof(mpi_backend)}(comm, mpi_backend)
    end
end
