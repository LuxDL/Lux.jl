using Lux, Test

include("../test_utils.jl")

rng = get_stable_rng(12345)

@testset "$mode: Incorrect " for (mode, aType, device, ongpu) in MODES
end
