module EnzymeExt

using LuxLib: Utils
using Static: True

Utils.is_extension_loaded(::Val{:Enzyme}) = True()

end
