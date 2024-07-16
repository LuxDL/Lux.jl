module DeviceUtilscuDNNExt

using CUDA: CUDA
using cuDNN: cuDNN
using DeviceUtils: DeviceUtils, CUDADevice, reset_gpu_device!

__init__() = reset_gpu_device!()

const USE_CUDA_GPU = Ref{Union{Nothing, Bool}}(nothing)

function _check_use_cuda!()
    USE_CUDA_GPU[] === nothing || return

    USE_CUDA_GPU[] = CUDA.functional()
    if USE_CUDA_GPU[]
        if !cuDNN.has_cudnn()
            @warn """
            cuDNN is not functional. Some functionality will not be available.
            """ maxlog=1

            # We make the device selectable only if cuDNN is functional
            # to avoid issues with convolutions and other deep learning operations
            USE_CUDA_GPU[] = false
        end
    end
    return
end

DeviceUtils.loaded(::Union{CUDADevice, Type{<:CUDADevice}}) = true

function DeviceUtils.functional(::Union{CUDADevice, Type{<:CUDADevice}})::Bool
    _check_use_cuda!()
    return USE_CUDA_GPU[]
end

end
