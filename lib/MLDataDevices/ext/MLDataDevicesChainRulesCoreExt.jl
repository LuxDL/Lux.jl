module MLDataDevicesChainRulesCoreExt

using Adapt: Adapt
using ChainRulesCore: ChainRulesCore, NoTangent, @non_differentiable

using MLDataDevices: AbstractDevice, UnknownDevice, get_device, get_device_type

@non_differentiable get_device(::Any)
@non_differentiable get_device_type(::Any)

function ChainRulesCore.rrule(
        ::typeof(Adapt.adapt_storage), to::AbstractDevice, x::AbstractArray)
    ∇adapt_storage = let dev = get_device(x)
        if dev === nothing || dev isa UnknownDevice
            @warn "`get_device(::$(typeof(x)))` returned `$(dev)`." maxlog=1
            Δ -> (NoTangent(), NoTangent(), Δ)
        else
            Δ -> (NoTangent(), NoTangent(), dev(Δ))
        end
    end
    return Adapt.adapt_storage(to, x), ∇adapt_storage
end

end
