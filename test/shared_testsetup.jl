@testsetup module SharedTestSetup

include("setup_modes.jl")

import Reexport: @reexport

using Lux
@reexport using ComponentArrays, LuxCore, LuxLib, LuxTestUtils, Random, StableRNGs, Test,
                Zygote, Statistics
using LuxTestUtils: @jet, @test_gradients, check_approx

# Some Helper Functions
function get_default_rng(mode::String)
    dev = mode == "cpu" ? LuxCPUDevice() :
          mode == "cuda" ? LuxCUDADevice() : mode == "amdgpu" ? LuxAMDGPUDevice() : nothing
    rng = default_device_rng(dev)
    return rng isa TaskLocalRNG ? copy(rng) : deepcopy(rng)
end

export @jet, @test_gradients, check_approx
export BACKEND_GROUP, MODES, cpu_testing, cuda_testing, amdgpu_testing, get_default_rng,
       StableRNG

end
