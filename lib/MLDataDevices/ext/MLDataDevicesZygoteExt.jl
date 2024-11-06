module MLDataDevicesZygoteExt

using Adapt: Adapt
using MLDataDevices: CPUDevice, CUDADevice, AMDGPUDevice, MetalDevice, oneAPIDevice, ReactantDevice
using Zygote: OneElement

Adapt.adapt_storage(::CPUDevice, x::OneElement) = x

for Dev in (CUDADevice, AMDGPUDevice, MetalDevice, oneAPIDevice, ReactantDevice)
    # use `@eval` to avoid ambiguity with adapt_storage(::CUDADevice, ::AbstractArray)
    @eval Adapt.adapt_storage(to::$Dev, x::OneElement) = Adapt.adapt(to, collect(x))
end

for Dev in (CUDADevice, AMDGPUDevice)
    # use `@eval` to avoid ambiguity with adapt_storage(::CUDADevice{Nothing}, ::AbstractArray)
    @eval Adapt.adapt_storage(to::$Dev{Nothing}, x::OneElement) = Adapt.adapt(to, collect(x))
end

end

