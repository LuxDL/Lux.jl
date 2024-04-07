module LuxComponentArraysExt

using ComponentArrays: ComponentArrays, ComponentArray, FlatAxis
using Lux: Lux, DistributedUtils

# Empty NamedTuple: Hack to avoid breaking precompilation
function ComponentArrays.ComponentArray(data::Vector{Any}, axes::Tuple{FlatAxis})
    length(data) == 0 && return ComponentArray(Float32[], axes)
    return ComponentArray{Any, 1, typeof(data), typeof(axes)}(data, axes)
end

Lux.__named_tuple(ca::ComponentArray) = NamedTuple(ca)

# Distributed Functionality
function DistributedUtils.synchronize!!(
        backend::Lux.AbstractLuxDistributedBackend, ps::ComponentArray; root::Int=0)
    ps_synced = DistributedUtils.synchronize!!(backend, ComponentArrays.getdata(ps); root)
    return ComponentArray(ps_synced, ComponentArrays.getaxes(ps))
end

end
