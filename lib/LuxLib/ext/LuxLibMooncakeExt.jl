module LuxLibMooncakeExt

using LuxLib: LuxLib, Utils, Impl, Traits, GenericBroadcastOp
using MLDataDevices: get_device_type
using Mooncake: @mooncake_overlay
using Static: True

## For mooncake we are missing some rules. For now use the basic versions of the kernels
@mooncake_overlay LuxLib.internal_operation_mode(xs::Tuple) =
    LuxLib.GenericBroadcastOp{get_device_type(xs)}()

# Utils extensions
@mooncake_overlay Utils.within_autodiff(x) = True()

end
