module LuxTestUtils

using ArrayInterface: ArrayInterface
using ComponentArrays: ComponentArray, getdata, getaxes
using DispatchDoctor: allow_unstable
using Functors: Functors
using MLDataDevices: cpu_device, gpu_device, get_device, get_device_type, AbstractGPUDevice
using Optimisers: Optimisers
using Pkg: PackageSpec
using Test:
    Test,
    Error,
    Broken,
    Pass,
    Fail,
    get_testset,
    @testset,
    @test,
    @test_skip,
    @test_broken,
    eval_test,
    Threw,
    Returned

# Autodiff
using ADTypes:
    AutoEnzyme, AutoFiniteDiff, AutoTracker, AutoForwardDiff, AutoReverseDiff, AutoZygote
using ChainRulesCore: ChainRulesCore
using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff
using Zygote: Zygote

const CRC = ChainRulesCore
const FD = FiniteDiff

# Check if JET will work
try
    using JET: JET, JETTestFailure, get_reports, report_call, report_opt
    # XXX: In 1.11, JET leads to stack overflows
    global JET_TESTING_ENABLED = v"1.10-" ≤ VERSION < v"1.11-"
catch err
    @error "`JET.jl` did not successfully precompile on $(VERSION). All `@jet` tests will \
            be skipped." maxlog = 1 err = err
    global JET_TESTING_ENABLED = false
end

# Check if Enzyme will work
try
    using Enzyme: Enzyme
    __ftest(x) = x
    Enzyme.autodiff(Enzyme.Reverse, __ftest, Enzyme.Active, Enzyme.Active(2.0))
    global ENZYME_TESTING_ENABLED = Sys.islinux()
catch err
    global ENZYME_TESTING_ENABLED = false
end

include("package_install.jl")
include("test_softfail.jl")
include("autodiff.jl")
include("jet.jl")

include("utils.jl")

export AutoEnzyme, AutoFiniteDiff, AutoTracker, AutoForwardDiff, AutoReverseDiff, AutoZygote
export test_gradients, @test_gradients
export Constant
export @jet, jet_target_modules!
export @test_softfail

end
