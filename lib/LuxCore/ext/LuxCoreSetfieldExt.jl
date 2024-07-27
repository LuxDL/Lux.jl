module LuxCoreSetfieldExt

using LuxCore: LuxCore
using Setfield: Setfield

LuxCore._is_extension_loaded(::Val{:Setfield}) = true

LuxCore.__setfield(x, prop, val) = Setfield.set(x, Setfield.PropertyLens{prop}(), val)
LuxCore.__setfield(x, (prop, val)) = LuxCore.__setfield(x, prop, val)

end
