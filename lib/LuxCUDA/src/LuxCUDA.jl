module LuxCUDA

using Reexport

@reexport using CUDA, CUDAKernels, NNlibCUDA, cuDNN

const USE_CUDA_GPU = Ref{Union{Nothing, Bool}}(nothing)

function _check_use_cuda!()
    USE_CUDA_GPU[] === nothing || return

    USE_CUDA_GPU[] = CUDA.functional()
    if USE_CUDA_GPU[]
        if !cuDNN.has_cudnn()
            @warn """
            CUDA.jl found cuda, but did not find libcudnn. Some functionality will not be
            available.""" maxlog=1
        end
    else
        @info """
        The GPU function is being called but the GPU is not accessible. Defaulting back to
        the CPU. (No action is required if you want to run on the CPU).""" maxlog=1
    end

    return
end

"""
    functional()

Check if LuxCUDA is functional.
"""
function functional()::Bool
    _check_use_cuda!()
    return USE_CUDA_GPU[]
end

end
