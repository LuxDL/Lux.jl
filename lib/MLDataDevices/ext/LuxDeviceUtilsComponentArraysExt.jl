module LuxDeviceUtilsComponentArraysExt

# FIXME: Needs upstreaming
using Adapt, ComponentArrays

function Adapt.adapt_structure(to, ca::ComponentArray)
    return ComponentArray(adapt(to, getdata(ca)), getaxes(ca))
end

end
