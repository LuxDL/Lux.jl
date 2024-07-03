@testsetup module SharedTestSetup

using GPUArrays, GPUArraysCore, Random, StableRNGs

GPUArraysCore.allowscalar(false)

const BACKEND_GROUP = lowercase(get(ENV, "BACKEND_GROUP", "All"))

RNGS_ARRTYPES = []
if BACKEND_GROUP == "all" || BACKEND_GROUP == "cpu"
    append!(RNGS_ARRTYPES,
        [(StableRNG(12345), AbstractArray, true, "cpu"),
            (Random.GLOBAL_RNG, AbstractArray, true, "cpu")])
end
if BACKEND_GROUP == "all" || BACKEND_GROUP == "cuda"
    using CUDA
    append!(RNGS_ARRTYPES,
        [(CUDA.default_rng(), CuArray, true, "cuda"),
            (GPUArrays.default_rng(CuArray), CuArray, true, "cuda")])
end
if BACKEND_GROUP == "all" || BACKEND_GROUP == "amdgpu"
    using AMDGPU
    append!(RNGS_ARRTYPES,
        [(AMDGPU.rocrand_rng(), ROCArray, true, "amdgpu"),
            (AMDGPU.gpuarrays_rng(), ROCArray, true, "amdgpu")])
end
if BACKEND_GROUP == "all" || BACKEND_GROUP == "metal"
    using Metal
    push!(RNGS_ARRTYPES, (Metal.gpuarrays_rng(), MtlArray, false, "metal"))
end
if BACKEND_GROUP == "all" || BACKEND_GROUP == "oneapi"
    using oneAPI
    using oneAPI: oneL0

    supports_fp64 = oneL0.module_properties(first(oneAPI.devices())).fp64flags &
                    oneL0.ZE_DEVICE_MODULE_FLAG_FP64 == oneL0.ZE_DEVICE_MODULE_FLAG_FP64

    push!(RNGS_ARRTYPES, (oneAPI.gpuarrays_rng(), oneArray, supports_fp64, "oneapi"))
end

export StableRNG, RNGS_ARRTYPES, BACKEND_GROUP, GPUArrays

end
