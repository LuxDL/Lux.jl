module LuxLibMKLExt

using LuxLib: Utils
using Static: True

Utils.is_extension_loaded(::Val{:MKL}) = True()

end
