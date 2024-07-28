@testsetup module SharedTestSetup

include("setup_modes.jl")

import Reexport: @reexport

using Lux, Functors

# FIXME: remove after https://github.com/FluxML/Functors.jl/pull/87 is merged
(::Functors.FlattenWalk)(f::F, ::NamedTuple{()}) where {F} = []
(::Functors.FlattenWalk)(f::F, ::@NamedTuple{}) where {F} = []

@reexport using ComponentArrays, LuxCore, LuxLib, LuxTestUtils, Random, StableRNGs, Test,
                Zygote, Statistics, Enzyme
using LuxTestUtils: check_approx

LuxTestUtils.jet_target_modules!(["Lux", "LuxCore", "LuxLib"])

# Some Helper Functions
function get_default_rng(mode::String)
    dev = mode == "cpu" ? LuxCPUDevice() :
          mode == "cuda" ? LuxCUDADevice() : mode == "amdgpu" ? LuxAMDGPUDevice() : nothing
    rng = default_device_rng(dev)
    return rng isa TaskLocalRNG ? copy(rng) : deepcopy(rng)
end

maybe_rewrite_to_crosscor(layer) = layer
function maybe_rewrite_to_crosscor(layer::Conv{N, use_bias, M}) where {N, use_bias, M}
    return CrossCor{N, use_bias, M}(
        layer.activation, layer.in_chs, layer.out_chs, layer.kernel_size,
        layer.stride, layer.pad, layer.dilation, layer.init_weight, layer.init_bias)
end

function maybe_rewrite_to_crosscor(mode, model)
    mode != "amdgpu" && return model
    return fmap(maybe_rewrite_to_crosscor, model)
end

export BACKEND_GROUP, MODES, cpu_testing, cuda_testing, amdgpu_testing, get_default_rng,
       StableRNG, maybe_rewrite_to_crosscor, check_approx

end
