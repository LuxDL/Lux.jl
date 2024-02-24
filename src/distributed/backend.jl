abstract type AbstractLuxDistributedTrainingBackend end

struct NCCLBackend <: AbstractLuxDistributedTrainingBackend
    function NCCLBackend()
        if Base.get_extension(@__MODULE__, :LuxCUDANCCLExt) === nothing
            error("`NCCLBackend` requires `CUDA.jl` and `NCCL.jl` to be loaded")
        end
        return new()
    end
end
