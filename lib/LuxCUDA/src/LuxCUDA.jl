module LuxCUDA

using Reexport

@reexport using CUDA, CUDAKernels, NNlibCUDA, cuDNN

const USE_CUDA = Ref{Union{Nothing, Bool}}(nothing)

function check_use_cuda!()
    USE_CUDA[] === nothing || return

    USE_CUDA[] = CUDA.functional()
    if USE_CUDA[]
        if !cuDNN.has_cudnn()
            @warn """
            CUDA.jl found cuda, but did not find libcudnn. Some functionality will not be
            available.""" maxlog=1
        end
    else
        @info """
        The CUDA GPU function is being called but the GPU is not accessible.
        Defaulting back to the CPU. (No action is required if you want to run on the CPU).
        """ maxlog=1
    end
end

"""
    functional()

Check if LuxCUDA is functional.
"""
function functional()::Bool
    check_use_cuda!()
    return USE_CUDA[]
end

end
