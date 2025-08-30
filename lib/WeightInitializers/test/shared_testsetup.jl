@testsetup module SharedTestSetup

using GPUArrays, GPUArraysCore, Random, StableRNGs

GPUArraysCore.allowscalar(false)

const BACKEND_GROUP = lowercase(get(ENV, "BACKEND_GROUP", "All"))

RNGS_ARRTYPES = []
if BACKEND_GROUP == "all" || BACKEND_GROUP == "cpu"
    append!(
        RNGS_ARRTYPES,
        [
            (StableRNG(12345), AbstractArray, true, "cpu"),
            (Random.GLOBAL_RNG, AbstractArray, true, "cpu"),
        ],
    )
end
if BACKEND_GROUP == "all" || BACKEND_GROUP == "cuda"
    using CUDA
    if CUDA.functional()
        append!(RNGS_ARRTYPES, [
            (CUDA.default_rng(), CuArray, true, "cuda"),
            # See https://github.com/JuliaGPU/GPUArrays.jl/issues/614
            # The GPUArrays.RNG for CUDA is a bit fragile
            # (GPUArrays.default_rng(CuArray), CuArray, true, "cuda"),
        ])
    else
        @assert BACKEND_GROUP == "all" "Expected CUDA.functional() to be true"
    end
end
if BACKEND_GROUP == "all" || BACKEND_GROUP == "amdgpu"
    using AMDGPU
    if AMDGPU.functional()
        append!(RNGS_ARRTYPES, [
            (AMDGPU.rocrand_rng(), ROCArray, true, "amdgpu"),
            # See https://github.com/JuliaGPU/GPUArrays.jl/issues/614
            # The GPUArrays.RNG for AMDGPU is a bit fragile
            # (AMDGPU.gpuarrays_rng(), ROCArray, true, "amdgpu"),
        ])
    else
        @assert BACKEND_GROUP == "all" "Expected AMDGPU.functional() to be true"
    end
end
if BACKEND_GROUP == "all" || BACKEND_GROUP == "metal"
    using Metal
    if Metal.functional()
        push!(RNGS_ARRTYPES, (Metal.gpuarrays_rng(), MtlArray, false, "metal"))
    else
        @assert BACKEND_GROUP == "all" "Expected Metal.functional() to be true"
    end
end
if BACKEND_GROUP == "all" || BACKEND_GROUP == "oneapi"
    using oneAPI
    using oneAPI: oneL0

    if oneAPI.functional()
        supports_fp64 =
            oneL0.module_properties(first(oneAPI.devices())).fp64flags &
            oneL0.ZE_DEVICE_MODULE_FLAG_FP64 == oneL0.ZE_DEVICE_MODULE_FLAG_FP64

        push!(RNGS_ARRTYPES, (oneAPI.gpuarrays_rng(), oneArray, supports_fp64, "oneapi"))
    else
        @assert BACKEND_GROUP == "all" "Expected oneAPI.functional() to be true"
    end
end

export StableRNG, RNGS_ARRTYPES, BACKEND_GROUP, GPUArrays

end
