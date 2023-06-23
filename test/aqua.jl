using Aqua, ChainRulesCore, Lux, Test

@testset "All Tests (except Ambiguity)" begin
    Aqua.test_all(Lux; ambiguities=false)
end

@testset "Ambiguity Tests" begin
    # The exclusions are due to CRC.@nondifferentiable
    Aqua.test_ambiguities(Lux; exclude=[ChainRulesCore.frule, Core.kwcall])
end
