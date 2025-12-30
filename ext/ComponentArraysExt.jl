module ComponentArraysExt

using ComponentArrays: ComponentArrays, ComponentArray
using Lux: Lux, DistributedUtils
using ForwardDiff: ForwardDiff

# Distributed Functionality
function DistributedUtils.synchronize!!(
    backend::Lux.AbstractLuxDistributedBackend, ps::ComponentArray; root::Int=0
)
    ps_synced = DistributedUtils.synchronize!!(backend, ComponentArrays.getdata(ps); root)
    return ComponentArray(ps_synced, ComponentArrays.getaxes(ps))
end

@static if pkgversion(ForwardDiff) ≥ v"1.0.1"
    # Apply overloads for GPU arrays
    Lux.@define_forwarddiff_gpu_overloads ComponentArray
end

end
