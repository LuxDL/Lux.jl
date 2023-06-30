using Aqua, ChainRulesCore, LuxLib, Test

@testset "All Tests (except Ambiguity)" begin
    Aqua.test_all(LuxLib; ambiguities=false)
end

@testset "Ambiguity Tests" begin
    # The exclusions are due to CRC.@nondifferentiable
    Aqua.test_ambiguities(LuxLib; exclude=[ChainRulesCore.frule, Core.kwcall])
end
