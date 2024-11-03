module LuxLibBLISBLASExt

using LuxLib: Utils
using Static: True

Utils.is_extension_loaded(::Val{:BLISBLAS}) = True()

end
