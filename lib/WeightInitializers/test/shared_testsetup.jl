@testsetup module SharedTestSetup

using GPUArraysCore, Random, StableRNGs

GPUArraysCore.allowscalar(false)

const BACKEND_GROUP = lowercase(get(ENV, "BACKEND_GROUP", "All"))

RNGS_ARRTYPES = []
if BACKEND_GROUP == "all" || BACKEND_GROUP == "cpu"
    append!(RNGS_ARRTYPES,
        [(StableRNG(12345), AbstractArray), (Random.GLOBAL_RNG, AbstractArray)])
end
if BACKEND_GROUP == "all" || BACKEND_GROUP == "cuda"
    using CUDA, GPUArrays
    append!(RNGS_ARRTYPES,
        [(CUDA.default_rng(), CuArray), (GPUArrays.default_rng(CuArray), CuArray)])
end
if BACKEND_GROUP == "all" || BACKEND_GROUP == "amdgpu"
    using AMDGPU
    append!(RNGS_ARRTYPES,
        [(AMDGPU.rocrand_rng(), ROCArray), (AMDGPU.gpuarrays_rng(), ROCArray)])
end
if BACKEND_GROUP == "all" || BACKEND_GROUP == "metal"
    using Metal
    push!(RNGS_ARRTYPES, (Metal.gpuarrays_rng(), MtlArray))
end
if BACKEND_GROUP == "all" || BACKEND_GROUP == "oneapi"
    using oneAPI
    push!(RNGS_ARRTYPES, (oneAPI.gpuarrays_rng(), oneArray))
end

export StableRNG, RNGS_ARRTYPES, BACKEND_GROUP

end
