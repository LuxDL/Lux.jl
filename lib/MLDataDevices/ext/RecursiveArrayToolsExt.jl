module RecursiveArrayToolsExt

using Adapt: Adapt, adapt
using MLDataDevices: MLDataDevices, Internal, AbstractDevice
using RecursiveArrayTools: VectorOfArray, DiffEqArray

MLDataDevices.isleaf(::VectorOfArray) = true
MLDataDevices.isleaf(::DiffEqArray) = true

# We want to preserve the structure
function Adapt.adapt_structure(to::AbstractDevice, x::VectorOfArray)
    return VectorOfArray(map(Base.Fix1(adapt, to), x.u))
end

function Adapt.adapt_structure(to::AbstractDevice, x::DiffEqArray)
    # Don't move the `time` to the GPU
    return DiffEqArray(map(Base.Fix1(adapt, to), x.u), x.t)
end

for op in (:get_device, :get_device_type)
    @eval function Internal.$(op)(x::Union{VectorOfArray,DiffEqArray})
        length(x.u) == 0 && return $(op == :get_device ? nothing : Nothing)
        return mapreduce(Internal.$(op), Internal.combine_devices, x.u)
    end
end

end
