module LuxMPIExt

using Lux: MPIBackend, NCCLBackend, DistributedUtils, __unwrap_val, MPI_CUDA_AWARE,
           MPI_ROCM_AWARE
using LuxDeviceUtils: AbstractLuxDevice, LuxCUDADevice, LuxAMDGPUDevice, cpu_device,
                      set_device!, functional
using MPI: MPI

function DistributedUtils.force_initialize(
        ::Type{MPIBackend}; cuda_devices=nothing, amdgpu_devices=nothing,
        force_cuda::Bool=false, caller::String="", force_amdgpu::Bool=false) # Undocumented internal kwarg
    !MPI.Initialized() && MPI.Init()
    DistributedUtils.MPI_Initialized[] = true

    local_rank = MPI.Comm_rank(MPI.COMM_WORLD)

    if cuda_devices !== missing && functional(LuxCUDADevice)
        if cuda_devices === nothing
            set_device!(LuxCUDADevice, nothing, local_rank + 1)
        else
            set_device!(LuxCUDADevice, cuda_devices[local_rank + 1])
        end
    elseif force_cuda
        error(lazy"CUDA devices are not functional (or `LuxCUDA.jl` not loaded) and `force_cuda` is set to `true`. This is caused by backend: $(caller).")
    end

    if amdgpu_devices !== missing && functional(LuxAMDGPUDevice)
        if amdgpu_devices === nothing
            set_device!(LuxAMDGPUDevice, nothing, local_rank + 1)
        else
            set_device!(LuxAMDGPUDevice, amdgpu_devices[local_rank + 1])
        end
    elseif force_amdgpu
        error(lazy"AMDGPU devices are not functional (or `AMDGPU.jl` not loaded) and `force_amdgpu` is set to `true`. This is caused by backend: $(caller).")
    end

    return
end

function DistributedUtils.unsafe_get_distributed_backend(::Type{MPIBackend})
    return MPIBackend(MPI.COMM_WORLD)
end

DistributedUtils.local_rank(backend::MPIBackend) = MPI.Comm_rank(backend.comm)

DistributedUtils.total_workers(backend::MPIBackend) = MPI.Comm_size(backend.comm)

# Broadcast
function DistributedUtils.bcast_impl!(
        backend::MPIBackend, sendrecvbuf, ::AbstractLuxDevice; root=0)
    MPI.Bcast!(sendrecvbuf, backend.comm; root)
    return
end

function DistributedUtils.bcast_impl!(
        backend::MPIBackend, sendbuf, recvbuf, dev::AbstractLuxDevice; root=0)
    DistributedUtils.bcast_impl!(
        backend, ifelse(DistributedUtils.local_rank(backend) == root, sendbuf, recvbuf),
        dev; root)
    return
end

for (aware, dType) in ((MPI_CUDA_AWARE, LuxCUDADevice), (MPI_ROCM_AWARE, LuxAMDGPUDevice))
    if !aware
        @eval begin
            function DistributedUtils.bcast_impl!(
                    backend::MPIBackend, sendrecvbuf, ::$dType; root=0)
                cdev = cpu_device()
                sendrecvbufₙ = sendrecvbuf |> cdev
                DistributedUtils.bcast_impl!(backend, sendrecvbufₙ, cdev; root)
                copyto!(sendrecvbuf, sendrecvbufₙ)
                return
            end

            function DistributedUtils.bcast_impl!(
                    backend::MPIBackend, sendbuf, recvbuf, ::$dType; root=0)
                cdev = cpu_device()
                sendbufₙ = sendbuf |> cdev
                recvbufₙ = recvbuf |> cdev
                DistributedUtils.bcast_impl!(backend, sendbufₙ, recvbufₙ, cdev; root)
                copyto!(recvbuf, recvbufₙ)
                return
            end
        end
    end
end

# Allreduce
function DistributedUtils.allreduce_impl!(
        backend::MPIBackend, sendrecvbuf, op::F, ::AbstractLuxDevice) where {F}
    mpiop = ifelse(op === DistributedUtils.avg, +, op)
    MPI.Allreduce!(sendrecvbuf, mpiop, backend.comm)
    if op === DistributedUtils.avg
        sendrecvbuf ./= DistributedUtils.total_workers(backend)
    end
    return
end

function DistributedUtils.allreduce_impl!(
        backend::MPIBackend, sendbuf, recvbuf, op::F, ::AbstractLuxDevice) where {F}
    mpiop = ifelse(op === DistributedUtils.avg, +, op)
    MPI.Allreduce!(sendbuf, recvbuf, mpiop, backend.comm)
    if op === DistributedUtils.avg
        recvbuf ./= DistributedUtils.total_workers(backend)
    end
    return
end

for (aware, dType) in ((MPI_CUDA_AWARE, LuxCUDADevice), (MPI_ROCM_AWARE, LuxAMDGPUDevice))
    if !aware
        @eval begin
            function DistributedUtils.allreduce_impl!(
                    backend::MPIBackend, sendrecvbuf, op::F, ::$dType) where {F}
                cdev = cpu_device()
                sendrecvbufₙ = sendrecvbuf |> cdev
                DistributedUtils.allreduce_impl!(backend, sendrecvbufₙ, op, cdev)
                copyto!(sendrecvbuf, sendrecvbufₙ)
                return
            end

            function DistributedUtils.allreduce_impl!(
                    backend::MPIBackend, sendbuf, recvbuf, op::F, ::$dType) where {F}
                cdev = cpu_device()
                sendbufₙ = sendbuf |> cdev
                recvbufₙ = recvbuf |> cdev
                DistributedUtils.allreduce_impl!(backend, sendbufₙ, recvbufₙ, op, cdev)
                copyto!(recvbuf, recvbufₙ)
                return
            end
        end
    end
end

# Reduce
function DistributedUtils.reduce_impl!(
        backend::MPIBackend, sendrecvbuf, op::F, ::AbstractLuxDevice; root::Int) where {F}
    mpiop = ifelse(op === DistributedUtils.avg, +, op)
    MPI.Reduce!(sendrecvbuf, mpiop, backend.comm; root)
    if op === DistributedUtils.avg
        sendrecvbuf ./= DistributedUtils.total_workers(backend)
    end
    return
end

function DistributedUtils.reduce_impl!(backend::MPIBackend, sendbuf, recvbuf, op::F,
        ::AbstractLuxDevice; root::Int) where {F}
    mpiop = ifelse(op === DistributedUtils.avg, +, op)
    MPI.Reduce!(sendbuf, recvbuf, mpiop, backend.comm; root)
    if op === DistributedUtils.avg
        recvbuf ./= DistributedUtils.total_workers(backend)
    end
    return
end

for (aware, dType) in ((MPI_CUDA_AWARE, LuxCUDADevice), (MPI_ROCM_AWARE, LuxAMDGPUDevice))
    if !aware
        @eval begin
            function DistributedUtils.reduce_impl!(backend::MPIBackend, sendrecvbuf, op::F,
                    ::$dType; root::Int) where {F}
                cdev = cpu_device()
                sendrecvbufₙ = sendrecvbuf |> cdev
                DistributedUtils.reduce_impl!(backend, sendrecvbufₙ, op, cdev; root)
                copyto!(sendrecvbuf, sendrecvbufₙ)
                return
            end

            function DistributedUtils.reduce_impl!(backend::MPIBackend, sendbuf, recvbuf,
                    op::F, ::$dType; root::Int) where {F}
                cdev = cpu_device()
                sendbufₙ = sendbuf |> cdev
                recvbufₙ = recvbuf |> cdev
                DistributedUtils.reduce_impl!(backend, sendbufₙ, recvbufₙ, op, cdev; root)
                copyto!(recvbuf, recvbufₙ)
                return
            end
        end
    end
end

end
