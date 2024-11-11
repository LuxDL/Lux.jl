module MLDataDevicesZygoteExt

using Adapt: Adapt
using MLDataDevices: CPUDevice, CUDADevice, AMDGPUDevice, MetalDevice, oneAPIDevice,
                     ReactantDevice
using Zygote: OneElement

Adapt.adapt_structure(::CPUDevice, x::OneElement) = x

for dev in (CUDADevice, AMDGPUDevice, MetalDevice, oneAPIDevice, ReactantDevice,
    CUDADevice{Nothing}, AMDGPUDevice{Nothing})
    # use `@eval` to avoid ambiguity with adapt_storage(::CUDADevice, ::AbstractArray)
    @eval Adapt.adapt_structure(to::$(dev), x::OneElement) = Adapt.adapt(to, collect(x))
end

end
