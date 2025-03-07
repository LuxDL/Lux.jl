@testsetup module SharedTestSetup

include("setup_modes.jl")

import Reexport: @reexport

using Lux, Functors
using Setfield: @set
using DispatchDoctor: allow_unstable
@reexport using ComponentArrays, LuxCore, LuxLib, LuxTestUtils, Random, StableRNGs, Test,
    Zygote, Statistics, Enzyme, LinearAlgebra, ForwardDiff
using MLDataDevices: default_device_rng, CPUDevice, CUDADevice, AMDGPUDevice
using LuxTestUtils: check_approx
using Static: True

using Octavian, LoopVectorization

LuxTestUtils.jet_target_modules!(["Lux", "LuxCore", "LuxLib"])
LinearAlgebra.BLAS.set_num_threads(Threads.nthreads())

# Some Helper Functions
function get_default_rng(mode::String)
    dev = mode == "cpu" ? CPUDevice() :
        mode == "cuda" ? CUDADevice() : mode == "amdgpu" ? AMDGPUDevice() : nothing
    rng = default_device_rng(dev)
    return rng isa TaskLocalRNG ? copy(rng) : deepcopy(rng)
end

maybe_rewrite_to_crosscor(layer) = layer
function maybe_rewrite_to_crosscor(layer::Conv)
    @set layer.cross_correlation = True()
    return layer
end

function maybe_rewrite_to_crosscor(mode, model)
    mode != "amdgpu" && return model
    return fmap(maybe_rewrite_to_crosscor, model)
end

sumabs2first(layer, x, ps, st) = sum(abs2, first(layer(x, ps, st)))
sumsumfirst(layer, x, ps, st) = sum(sum, first(layer(x, ps, st)))

export BACKEND_GROUP, MODES, cpu_testing, cuda_testing, amdgpu_testing, get_default_rng,
    StableRNG, maybe_rewrite_to_crosscor, check_approx, allow_unstable,
    sumabs2first, sumsumfirst

end
