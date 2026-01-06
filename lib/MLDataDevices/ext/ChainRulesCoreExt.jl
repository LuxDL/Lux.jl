module ChainRulesCoreExt

using Adapt: Adapt
using ChainRulesCore: ChainRulesCore, NoTangent, ProjectTo, @non_differentiable

using MLDataDevices:
    AbstractDevice,
    MLDataDevices,
    UnknownDevice,
    get_device,
    get_device_type,
    reactant_device,
    cpu_device,
    gpu_device

@non_differentiable get_device(::Any...)
@non_differentiable get_device_type(::Any...)
@non_differentiable gpu_device(::Any...)
@non_differentiable cpu_device(::Any...)
@non_differentiable reactant_device(::Any...)

function ChainRulesCore.rrule(
    ::typeof(Adapt.adapt_storage), to::AbstractDevice, x::AbstractArray
)
    dev = get_device(x)
    y = Adapt.adapt_storage(to, x)
    if dev === nothing || dev isa UnknownDevice
        dev isa UnknownDevice &&
            @warn "`get_device(::$(typeof(x)))` returned `$(dev)`." maxlog = 1
        ∇adapt_storage_unknown = Δ -> (NoTangent(), NoTangent(), Δ)
        return y, ∇adapt_storage_unknown
    else
        ∇adapt_storage = let dev = dev, x = x
            Δ -> begin
                ∂x = ChainRulesCore.@thunk ProjectTo(x)(dev(ChainRulesCore.unthunk(Δ)))
                return NoTangent(), NoTangent(), ∂x
            end
        end
        return Adapt.adapt_storage(to, x), ∇adapt_storage
    end
end

MLDataDevices.isleaf(::ChainRulesCore.AbstractTangent) = true
MLDataDevices.isleaf(::ChainRulesCore.AbstractThunk) = true

end
