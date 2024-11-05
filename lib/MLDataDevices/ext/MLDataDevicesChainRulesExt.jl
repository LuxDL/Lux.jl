module MLDataDevicesChainRulesExt

using Adapt: Adapt
using ChainRules: OneElement
using MLDataDevices: CPUDevice, CUDADevice, AMDGPUDevice, MetalDevice, ReactantDevice

Adapt.adapt_storage(::CPUDevice, x::OneElement) = x
for Dev in (CUDADevice, AMDGPUDevice, MetalDevice, ReactantDevice)
    # use `@eval` to avoid ambiguity with adapt_storage(::CUDADevice, ::AbstractArray)
    @eval Adapt.adapt_storage(to::$Dev, x::OneElement) = Adapt.adapt(to, collect(x))
end

end