module LuxLibAppleAccelerateExt

using LuxLib: Utils
using Static: True

Utils.is_extension_loaded(::Val{:AppleAccelerate}) = True()

end
