# Load all trigger packages
import LuxAMDGPU, LuxCUDA, FillArrays, Metal, RecursiveArrayTools, SparseArrays, Zygote,
       oneAPI
using ExplicitImports, LuxDeviceUtils

@test check_no_implicit_imports(LuxDeviceUtils) === nothing
@test check_no_stale_explicit_imports(LuxDeviceUtils) === nothing
