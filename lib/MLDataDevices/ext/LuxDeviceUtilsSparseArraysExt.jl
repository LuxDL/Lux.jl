module LuxDeviceUtilsSparseArraysExt

import Adapt: adapt_storage
import LuxDeviceUtils: LuxCPUAdaptor
import SparseArrays: AbstractSparseArray

adapt_storage(::LuxCPUAdaptor, x::AbstractSparseArray) = x

end
