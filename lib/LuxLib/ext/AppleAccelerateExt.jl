module AppleAccelerateExt

using LuxLib.Utils: Utils, True

Utils.is_extension_loaded(::Val{:AppleAccelerate}) = True()

end
