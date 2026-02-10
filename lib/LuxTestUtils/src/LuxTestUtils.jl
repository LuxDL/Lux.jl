module LuxTestUtils

using Adapt: Adapt, adapt
using ArrayInterface: ArrayInterface
using ComponentArrays: ComponentArray, getdata, getaxes
using DispatchDoctor: allow_unstable
using Functors: Functors, fmap
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
    AutoEnzyme, AutoFiniteDiff, AutoTracker, AutoForwardDiff, AutoReverseDiff, AutoZygote, AutoMooncake
using ChainRulesCore: ChainRulesCore
using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff
using Zygote: Zygote
using Mooncake

const CRC = ChainRulesCore
const FD = FiniteDiff

const JET_TESTING_ENABLED = Ref{Bool}(false)
const ENZYME_TESTING_ENABLED = Ref{Bool}(false)
const ZYGOTE_TESTING_ENABLED = Ref{Bool}(false)

# Check if JET will work
try
    using JET: JET, JETTestFailure, get_reports, report_call, report_opt
    JET_TESTING_ENABLED[] = true
catch err
    @error "`JET.jl` did not successfully precompile on $(VERSION). All `@jet` tests will \
            be skipped." maxlog = 1 err = err
    JET_TESTING_ENABLED[] = false
end

# Check if Enzyme will work (only on non-prerelease versions)
@static if isempty(VERSION.prerelease)
    try
        using Enzyme: Enzyme
        Enzyme.gradient(Enzyme.Reverse, Base.Fix1(sum, abs2), ones(Float32, 10))
        ENZYME_TESTING_ENABLED[] = Sys.islinux()
    catch err
        @error "`Enzyme.jl` did not successfully differentiate a simple function or \
                failed to load on $(VERSION). All Enzyme tests will be \
                skipped." maxlog = 1 err = err
        ENZYME_TESTING_ENABLED[] = false
    end
end

function __init__()
    ZYGOTE_TESTING_ENABLED[] = VERSION < v"1.12-"

    if JET_TESTING_ENABLED[]
        # JET doesn't work nicely on 1.11
        JET_TESTING_ENABLED[] = VERSION < v"1.11-" || VERSION ≥ v"1.12-"
    end
end

include("package_install.jl")
include("test_softfail.jl")
include("autodiff.jl")
include("jet.jl")

include("utils.jl")

export AutoEnzyme, AutoFiniteDiff, AutoTracker, AutoForwardDiff, AutoReverseDiff, AutoZygote, AutoMooncake
export test_gradients, @test_gradients
export Constant
export @jet, jet_target_modules!
export @test_softfail

end
