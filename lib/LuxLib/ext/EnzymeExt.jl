module EnzymeExt

using LuxLib.Utils: Utils, True

Utils.is_extension_loaded(::Val{:Enzyme}) = True()

end
