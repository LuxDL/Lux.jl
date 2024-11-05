module LuxTestUtils

using ArrayInterface: ArrayInterface
using ComponentArrays: ComponentArray, getdata, getaxes
using DispatchDoctor: allow_unstable
using Functors: Functors
using MLDataDevices: cpu_device, gpu_device, get_device, get_device_type, AbstractGPUDevice
using Test: Test, Error, Broken, Pass, Fail, get_testset, @testset, @test, @test_skip,
            @test_broken, eval_test, Threw, Returned

# Autodiff
using ADTypes: AutoEnzyme, AutoFiniteDiff, AutoTracker, AutoForwardDiff, AutoReverseDiff,
               AutoZygote
using ChainRulesCore: ChainRulesCore
using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Tracker: Tracker
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
            be skipped." maxlog=1 err=err
    global JET_TESTING_ENABLED = false
end

# Check if Enzyme will work
try
    using Enzyme: Enzyme
    __ftest(x) = x
    Enzyme.autodiff(Enzyme.Reverse, __ftest, Enzyme.Active, Enzyme.Active(2.0))
    # XXX: Enzyme has been causing some issues lately. Let's just disable it for now.
    #      We still have opt-in testing available for Enzyme.
    # XXX: Lift this once Enzyme supports 1.11 properly
    global ENZYME_TESTING_ENABLED = false # v"1.10-" ≤ VERSION < v"1.11-"
catch err
    global ENZYME_TESTING_ENABLED = false
end

include("test_softfail.jl")
include("utils.jl")
include("autodiff.jl")
include("jet.jl")

export AutoEnzyme, AutoFiniteDiff, AutoTracker, AutoForwardDiff, AutoReverseDiff, AutoZygote
export test_gradients, @test_gradients
export @jet, jet_target_modules!
export @test_softfail

end
