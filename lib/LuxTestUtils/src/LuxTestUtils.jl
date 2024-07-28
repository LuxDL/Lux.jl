module LuxTestUtils

using ComponentArrays: ComponentArray, getdata, getaxes
using DispatchDoctor: allow_unstable
using Functors: Functors
using MLDataDevices: cpu_device, gpu_device, get_device, get_device_type, AbstractGPUDevice
using Test: Test, Error, Broken, Pass, Fail, get_testset, @testset, @test, @test_skip,
            @test_broken

# Autodiff
using ADTypes: AutoEnzyme, AutoFiniteDiff, AutoTracker, AutoForwardDiff, AutoReverseDiff,
               AutoZygote
using ChainRulesCore: ChainRulesCore
using Enzyme: Enzyme
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
    global JET_TESTING_ENABLED = true
catch err
    @error "`JET.jl` did not successfully precompile on $(VERSION). All `@jet` tests will \
            be skipped." maxlog=1 err=err
    global JET_TESTING_ENABLED = false
end

# Check if Enzyme will work
try
    __ftest(x) = x
    Enzyme.autodiff(Enzyme.Reverse, __ftest, Enzyme.Active, Enzyme.Active(2.0))
    global ENZYME_TESTING_ENABLED = true
catch err
    @error "`Enzyme.jl` is currently not functional on $(VERSION). Enzyme tests will be \
            skipped." maxlog=1 err=err
    global ENZYME_TESTING_ENABLED = false
end

include("utils.jl")
include("autodiff.jl")
include("jet.jl")

export AutoEnzyme, AutoFiniteDiff, AutoTracker, AutoForwardDiff, AutoReverseDiff,
       AutoZygote
export test_gradients
export @jet, jet_target_modules!

end
