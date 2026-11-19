module MKLExt

using LuxLib.Utils: Utils, True

Utils.is_extension_loaded(::Val{:MKL}) = True()

end
