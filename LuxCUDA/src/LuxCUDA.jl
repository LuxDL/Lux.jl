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
            cuDNN is not functional in CUDA.jl. Some functionality will not be available.
            """ maxlog=1
        end
    else
        @warn "LuxCUDA is loaded but the CUDA GPU is not functional." maxlog=1
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
