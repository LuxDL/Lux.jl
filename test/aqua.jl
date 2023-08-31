using Aqua, ChainRulesCore, Lux, Test

@testset "All Tests (except Ambiguity & Piracy)" begin
    Aqua.test_all(Lux; ambiguities=false)
end

@testset "Ambiguity Tests" begin
    # The exclusions are due to CRC.@non_differentiable
    Aqua.test_ambiguities(Lux; exclude=[ChainRulesCore.frule, Core.kwcall])
end

@testset "Piracy Tests" begin
    # The exclusions are due to CRC.@non_differentiable
    Aqua.test_piracy(Lux;
        treat_as_own=[ChainRulesCore.frule, ChainRulesCore.rrule, Core.kwcall])
end
