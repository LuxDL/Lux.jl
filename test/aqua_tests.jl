@testitem "Aqua: Quality Assurance" begin
    using Aqua, ChainRulesCore

    Aqua.test_all(Lux; piracies=false)
    Aqua.test_piracies(Lux;
        treat_as_own=[ChainRulesCore.frule, ChainRulesCore.rrule, Core.kwcall])
end
