module MLDataDevicesChainRulesCoreExt

using Adapt: Adapt
using ChainRulesCore: ChainRulesCore, NoTangent, ProjectTo, @non_differentiable

using MLDataDevices: AbstractDevice, UnknownDevice, get_device, get_device_type

@non_differentiable get_device(::Any)
@non_differentiable get_device_type(::Any)

function ChainRulesCore.rrule(::typeof(Adapt.adapt), to::AbstractDevice, x::AbstractArray)
    dev = get_device(x)
    y = Adapt.adapt_storage(to, x)
    if dev === nothing || dev isa UnknownDevice
        dev isa UnknownDevice &&
            @warn "`get_device(::$(typeof(x)))` returned `$(dev)`." maxlog=1
        ∇adapt_storage_unknown = Δ -> (NoTangent(), NoTangent(), Δ)
        return y, ∇adapt_storage_unknown
    else
        ∇adapt_storage = let dev = dev, x = x
            Δ -> (NoTangent(), NoTangent(), ProjectTo(x)(dev(Δ)))
        end
        return Adapt.adapt_storage(to, x), ∇adapt_storage
    end
end

end
