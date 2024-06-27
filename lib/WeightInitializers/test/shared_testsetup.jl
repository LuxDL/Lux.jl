@testsetup module SharedTestSetup

using CUDA, Random, StableRNGs

CUDA.allowscalar(false)

const BACKEND_GROUP = lowercase(get(ENV, "BACKEND_GROUP", "All"))

RNGS_ARRTYPES = []
if BACKEND_GROUP == "all" || BACKEND_GROUP == "cpu"
    append!(RNGS_ARRTYPES,
        [(StableRNG(12345), AbstractArray), (Random.GLOBAL_RNG, AbstractArray)])
end
if BACKEND_GROUP == "all" || BACKEND_GROUP == "cuda"
    push!(RNGS_ARRTYPES, (CUDA.default_rng(), CuArray))
end

export StableRNG, RNGS_ARRTYPES

end
