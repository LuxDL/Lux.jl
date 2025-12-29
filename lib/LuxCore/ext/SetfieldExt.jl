module SetfieldExt

using LuxCore: LuxCore
using Setfield: Setfield

LuxCore.Internal.is_extension_loaded(::Val{:Setfield}) = true

function LuxCore.Internal.setfield_impl(x, prop, val)
    return Setfield.set(x, Setfield.PropertyLens{prop}(), val)
end
function LuxCore.Internal.setfield_impl(x, (prop, val))
    return LuxCore.Internal.setfield_impl(x, prop, val)
end

end
