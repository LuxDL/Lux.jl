module MLDataDevicesChainRulesExt

using Adapt: Adapt
using MLDataDevices: CPUDevice, CUDADevice, AMDGPUDevice, MetalDevice,OneAPIDevice, ReactantDevice
using ChainRules: OneElement

Adapt.adapt_storage(::CPUDevice, x::OneElement) = x

for Dev in (CUDADevice, AMDGPUDevice, MetalDevice, OneAPIDevice, ReactantDevice)
    # use `@eval` to avoid ambiguity with adapt_storage(::CUDADevice, ::AbstractArray)
    @eval Adapt.adapt_storage(to::$Dev, x::OneElement) = Adapt.adapt(to, collect(x))
end

for Dev in (CUDADevice, AMDGPUDevice)
    # use `@eval` to avoid ambiguity with adapt_storage(::CUDADevice{Nothing}, ::AbstractArray)
    @eval Adapt.adapt_storage(to::$Dev, x::OneElement) = Adapt.adapt(to, collect(x))
end

end
