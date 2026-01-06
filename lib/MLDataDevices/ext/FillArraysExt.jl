module FillArraysExt

using Adapt: Adapt
using FillArrays: AbstractFill, OneElement, Fill, Ones, Zeros
using MLDataDevices: CPUDevice, ReactantDevice, AbstractDevice, Internal

Adapt.adapt_structure(::CPUDevice, x::AbstractFill) = x
function Adapt.adapt_structure(dev::CPUDevice, x::Ones{T}) where {T}
    return Ones{Adapt.adapt(dev, T)}(axes(x))
end
function Adapt.adapt_structure(dev::CPUDevice, x::Zeros{T}) where {T}
    return Zeros{Adapt.adapt(dev, T)}(axes(x))
end
Adapt.adapt_structure(dev::CPUDevice, x::Fill) = Fill(Adapt.adapt(dev, x.value), axes(x))
function Adapt.adapt_structure(dev::CPUDevice, x::OneElement)
    return OneElement(Adapt.adapt(dev, x.val), x.ind, x.axes)
end

Adapt.adapt_structure(dev::ReactantDevice, x::AbstractFill) = Internal.to_rarray(dev, x)
Adapt.adapt_structure(dev::ReactantDevice, x::OneElement) = Internal.to_rarray(dev, x)

Adapt.adapt_structure(to::AbstractDevice, x::AbstractFill) = Adapt.adapt(to, collect(x))
Adapt.adapt_structure(to::AbstractDevice, x::OneElement) = Adapt.adapt(to, collect(x))

Internal.get_device(::AbstractFill{T}) where {T} = Internal.get_device(T)
Internal.get_device(f::Fill) = Internal.get_device(f.value)
Internal.get_device(e::OneElement) = Internal.get_device(e.val)

Internal.get_device_type(::AbstractFill{T}) where {T} = Internal.get_device_type(T)
Internal.get_device_type(f::Fill) = Internal.get_device_type(f.value)
Internal.get_device_type(e::OneElement) = Internal.get_device_type(e.val)

end
