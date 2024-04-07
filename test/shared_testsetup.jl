@testsetup module SharedTestSetup
import Reexport: @reexport

using Lux
@reexport using ComponentArrays, LuxCore, LuxLib, LuxTestUtils, Random, StableRNGs, Test,
                Zygote, Statistics
using LuxTestUtils: @jet, @test_gradients, check_approx

include("setup_modes.jl")

# Some Helper Functions
function get_default_rng(mode::String)
    dev = mode == "CPU" ? LuxCPUDevice() :
          mode == "CUDA" ? LuxCUDADevice() : mode == "AMDGPU" ? LuxAMDGPUDevice() : nothing
    rng = default_device_rng(dev)
    return rng isa TaskLocalRNG ? copy(rng) : deepcopy(rng)
end

get_stable_rng(seed=12345) = StableRNG(seed)

__display(args...) = (println(); display(args...))

# AMDGPU Specifics
function _rocRAND_functional()
    try
        get_default_rng("AMDGPU")
        return true
    catch
        return false
    end
end

export @jet, @test_gradients, check_approx
export GROUP, MODES, cpu_testing, cuda_testing, amdgpu_testing, get_default_rng,
       get_stable_rng, __display, _rocRAND_functional

end
