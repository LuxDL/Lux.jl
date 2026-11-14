module BLISBLASExt

using LuxLib.Utils: Utils, True

Utils.is_extension_loaded(::Val{:BLISBLAS}) = True()

end
