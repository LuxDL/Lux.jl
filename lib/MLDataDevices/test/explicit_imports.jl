# Load all trigger packages
import FillArrays, RecursiveArrayTools, SparseArrays, Zygote
using ExplicitImports, LuxDeviceUtils

@test check_no_implicit_imports(LuxDeviceUtils) === nothing
@test check_no_stale_explicit_imports(LuxDeviceUtils) === nothing
