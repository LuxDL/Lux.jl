abstract type AbstractLuxDistributedBackend end

struct NCCLBackend{C} <: AbstractLuxDistributedBackend
    comm::C

    function NCCLBackend(comm=nothing)
        if Base.get_extension(@__MODULE__, :LuxCUDAMPINCCLExt) === nothing
            error("`NCCLBackend` requires `CUDA.jl`, `MPI.jl` and `NCCL.jl` to be loaded.")
        end
        return new{typeof(comm)}(comm)
    end
end

struct MPIBackend{C} <: AbstractLuxDistributedBackend
    comm::C

    function MPIBackend(comm=nothing)
        if Base.get_extension(@__MODULE__, :LuxMPIExt) === nothing
            error("`MPIBackend` requires `MPI.jl` to be loaded.")
        end
        return new{typeof(comm)}(comm)
    end
end
