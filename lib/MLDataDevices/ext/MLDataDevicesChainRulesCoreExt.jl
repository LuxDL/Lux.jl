module MLDataDevicesChainRulesCoreExt

using Adapt: Adapt
using ChainRulesCore: ChainRulesCore, NoTangent, @non_differentiable

using MLDataDevices: AbstractDevice, get_device, get_device_type

@non_differentiable get_device(::Any)
@non_differentiable get_device_type(::Any)

function ChainRulesCore.rrule(
        ::typeof(Adapt.adapt_storage), to::AbstractDevice, x::AbstractArray)
    ∇adapt_storage = let x = x
        Δ -> (NoTangent(), NoTangent(), (get_device(x))(Δ))
    end
    return Adapt.adapt_storage(to, x), ∇adapt_storage
end

end
