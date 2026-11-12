using Lux, Functors
using DispatchDoctor: allow_unstable
using ComponentArrays,
    LuxCore,
    LuxLib,
    LuxTestUtils,
    Random,
    StableRNGs,
    Test,
    Zygote,
    Statistics,
    Enzyme,
    LinearAlgebra,
    ForwardDiff
using MLDataDevices: default_device_rng, CPUDevice, CUDADevice
using LuxTestUtils: LuxTestUtils, check_approx

using Octavian, LoopVectorization

include("setup_modes.jl")

LuxTestUtils.jet_target_modules!(["Lux", "LuxCore", "LuxLib"])
LinearAlgebra.BLAS.set_num_threads(Threads.nthreads())

# Some Helper Functions
function get_default_rng(mode::String)
    dev = if mode == "cpu"
        CPUDevice()
    elseif mode == "cuda"
        CUDADevice()
    else
        nothing
    end
    rng = default_device_rng(dev)
    return rng isa TaskLocalRNG ? copy(rng) : deepcopy(rng)
end

sumabs2first(layer, x, ps, st) = sum(abs2, first(layer(x, ps, st)))
sumsumfirst(layer, x, ps, st) = sum(sum, first(layer(x, ps, st)))
