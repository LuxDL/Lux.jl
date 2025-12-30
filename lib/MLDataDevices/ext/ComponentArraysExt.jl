module ComponentArraysExt

using ComponentArrays: ComponentArrays
using MLDataDevices: MLDataDevices

MLDataDevices.isleaf(::ComponentArrays.ComponentArray) = true

end
