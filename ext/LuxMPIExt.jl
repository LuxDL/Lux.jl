module LuxMPIExt

using Lux: MPIBackend, NCCLBackend, DistributedUtils, __is_extension_loaded, __unwrap_val,
           MPI_CUDA_AWARE, MPI_ROCM_AWARE
using LuxDeviceUtils: AbstractLuxDevice, LuxCUDADevice, LuxAMDGPUDevice, cpu_device,
                      set_device!, __is_functional
using MPI: MPI

function DistributedUtils.__initialize(
        ::Val{:MPI}; cuda_devices=nothing, amdgpu_devices=nothing)
    !MPI.Initialized() && MPI.Init()
    DistributedUtils.MPI_Initialized[] = true

    local_rank = MPI.Comm_rank(MPI.COMM_WORLD)

    if cuda_devices !== missing && __is_functional(LuxCUDADevice)
        if cuda_devices === nothing
            set_device!(LuxCUDADevice, nothing, local_rank)
        else
            set_device!(LuxCUDADevice, cuda_devices[local_rank])
        end
    end

    if amdgpu_devices !== missing && __is_functional(LuxAMDGPUDevice)
        if amdgpu_devices === nothing
            set_device!(LuxAMDGPUDevice, nothing, local_rank)
        else
            set_device!(LuxAMDGPUDevice, amdgpu_devices[local_rank])
        end
    end

    return
end

DistributedUtils.__get_distributed_backend(::Val{:MPI}) = MPIBackend(MPI.COMM_WORLD)

DistributedUtils.local_rank(backend::MPIBackend) = MPI.Comm_rank(backend.comm)

DistributedUtils.total_workers(backend::MPIBackend) = MPI.Comm_size(backend.comm)

# Broadcast

function DistributedUtils.__bcast!(
        backend::MPIBackend, sendrecvbuf, dev::AbstractLuxDevice; root=0)
    MPI.Bcast!(sendrecvbuf, backend.comm; root)
    return sendrecvbuf
end

function DistributedUtils.__bcast!(
        backend::MPIBackend, sendbuf, recvbuf, dev::AbstractLuxDevice; root=0)
    MPI.Bcast!(sendbuf, recvbuf, backend.comm; root)
    return recvbuf
end

for (aware, dType) in ((MPI_CUDA_AWARE, LuxCUDADevice), (MPI_ROCM_AWARE, LuxAMDGPUDevice))
    if !aware
        @eval begin
            function DistributedUtils.__bcast!(
                    backend::MPIBackend, sendrecvbuf, dev::$dType; root=0)
                cdev = cpu_device()
                sendrecvbuf_ = sendrecvbuf |> cdev
                DistributedUtils.__bcast!(backend, sendrecvbuf_, cdev; root)
                sendrecvbuf .= dev(sendrecvbuf_)
                return sendrecvbuf
            end

            function DistributedUtils.__bcast!(
                    backend::MPIBackend, sendbuf, recvbuf, dev::$dType; root=0)
                cdev = cpu_device()
                sendbuf_ = sendbuf |> cdev
                recvbuf_ = recvbuf |> cdev
                DistributedUtils.__bcast!(backend, sendbuf_, recvbuf_, cdev; root)
                recvbuf .= dev(recvbuf_)
                return recvbuf
            end
        end
    end
end

# Allreduce

function DistributedUtils.__allreduce!(
        backend::MPIBackend, sendrecvbuf, op::F, dev::AbstractLuxDevice) where {F}
    MPI.Allreduce!(sendrecvbuf, op, backend.comm)
    if op === typeof(DistributedUtils.avg)
        sendrecvbuf ./= DistributedUtils.total_workers(backend)
    end
    return sendrecvbuf
end

function DistributedUtils.__allreduce!(
        backend::MPIBackend, sendbuf, recvbuf, op::F, dev::AbstractLuxDevice) where {F}
    MPI.Allreduce!(sendbuf, recvbuf, op, backend.comm)
    if op === typeof(DistributedUtils.avg)
        recvbuf ./= DistributedUtils.total_workers(backend)
    end
    return recvbuf
end

for (aware, dType) in ((MPI_CUDA_AWARE, LuxCUDADevice), (MPI_ROCM_AWARE, LuxAMDGPUDevice))
    if !aware
        @eval begin
            function DistributedUtils.__allreduce!(
                    backend::MPIBackend, sendrecvbuf, op::F, dev::$dType) where {F}
                cdev = cpu_device()
                sendrecvbuf_ = sendrecvbuf |> cdev
                DistributedUtils.__allreduce!(backend, sendrecvbuf_, op, cdev)
                sendrecvbuf .= dev(sendrecvbuf_)
                return sendrecvbuf
            end

            function DistributedUtils.__allreduce!(
                    backend::MPIBackend, sendbuf, recvbuf, op::F, dev::$dType) where {F}
                cdev = cpu_device()
                sendbuf_ = sendbuf |> cdev
                recvbuf_ = recvbuf |> cdev
                DistributedUtils.__allreduce!(backend, sendbuf_, recvbuf_, op, cdev)
                recvbuf .= dev(recvbuf_)
                return recvbuf
            end
        end
    end
end

# Reduce

function DistributedUtils.__reduce!(backend::MPIBackend, sendrecvbuf, op::F,
        dev::AbstractLuxDevice; root::Int) where {F}
    MPI.Reduce!(sendrecvbuf, op, backend.comm; root)
    if op === typeof(DistributedUtils.avg)
        sendrecvbuf ./= DistributedUtils.total_workers(backend)
    end
    return sendrecvbuf
end

function DistributedUtils.__reduce!(backend::MPIBackend, sendbuf, recvbuf, op::F,
        dev::AbstractLuxDevice; root::Int) where {F}
    MPI.Reduce!(sendbuf, recvbuf, op, backend.comm; root)
    if op === typeof(DistributedUtils.avg)
        recvbuf ./= DistributedUtils.total_workers(backend)
    end
    return recvbuf
end

for (aware, dType) in ((MPI_CUDA_AWARE, LuxCUDADevice), (MPI_ROCM_AWARE, LuxAMDGPUDevice))
    if !aware
        @eval begin
            function DistributedUtils.__reduce!(backend::MPIBackend, sendrecvbuf, op::F,
                    dev::$dType; root::Int) where {F}
                cdev = cpu_device()
                sendrecvbuf_ = sendrecvbuf |> cdev
                DistributedUtils.__reduce!(backend, sendrecvbuf_, op, cdev; root)
                sendrecvbuf .= dev(sendrecvbuf_)
                return sendrecvbuf
            end

            function DistributedUtils.__reduce!(backend::MPIBackend, sendbuf, recvbuf,
                    op::F, dev::$dType; root::Int) where {F}
                cdev = cpu_device()
                sendbuf_ = sendbuf |> cdev
                recvbuf_ = recvbuf |> cdev
                DistributedUtils.__reduce!(backend, sendbuf_, recvbuf_, op, cdev; root)
                recvbuf .= dev(recvbuf_)
                return recvbuf
            end
        end
    end
end

end
