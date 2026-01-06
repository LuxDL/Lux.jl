abstract type AbstractLuxDistributedBackend end

"""
    MPIBackend(comm = nothing)

Create an MPI backend for distributed training. Users should not use this function directly.
Instead use [`DistributedUtils.get_distributed_backend(MPIBackend)`](@ref).
"""
struct MPIBackend{C} <: AbstractLuxDistributedBackend
    comm::C

    function MPIBackend(comm=nothing)
        if !is_extension_loaded(Val(:MPI))
            error("`MPIBackend` requires `MPI.jl` to be loaded.")
        end
        return new{typeof(comm)}(comm)
    end
end

"""
    NCCLBackend(comm = nothing, mpi_backend = nothing)

Create an NCCL backend for distributed training. Users should not use this function
directly. Instead use [`DistributedUtils.get_distributed_backend(NCCLBackend)`](@ref).
"""
struct NCCLBackend{C,M<:Union{Nothing,MPIBackend}} <: AbstractLuxDistributedBackend
    comm::C
    mpi_backend::M

    function NCCLBackend(comm=nothing, mpi_backend=nothing)
        if !is_extension_loaded(Val(:MPINCCL))
            error("`NCCLBackend` requires `CUDA.jl`, `MPI.jl` and `NCCL.jl` to be loaded.")
        end
        return new{typeof(comm),typeof(mpi_backend)}(comm, mpi_backend)
    end
end
