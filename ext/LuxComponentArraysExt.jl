module LuxComponentArraysExt

using ComponentArrays: ComponentArrays, ComponentArray
using Lux: Lux, DistributedUtils

# Distributed Functionality
function DistributedUtils.synchronize!!(
    backend::Lux.AbstractLuxDistributedBackend, ps::ComponentArray; root::Int=0
)
    ps_synced = DistributedUtils.synchronize!!(backend, ComponentArrays.getdata(ps); root)
    return ComponentArray(ps_synced, ComponentArrays.getaxes(ps))
end

end
