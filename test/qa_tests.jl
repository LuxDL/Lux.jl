@testitem "Aqua: Quality Assurance" begin
    using Aqua, ChainRulesCore

    Aqua.test_all(Lux; piracies=false)
    Aqua.test_piracies(
        Lux; treat_as_own=[ChainRulesCore.frule, ChainRulesCore.rrule, Core.kwcall])
end

@testitem "Explicit Imports: Quality Assurance" begin
    # Load all trigger packages
    import Lux, ComponentArrays, ReverseDiff, ChainRules, Flux, LuxAMDGPU, SimpleChains,
           Tracker, Zygote

    using ExplicitImports

    # Skip our own packages
    @test check_no_implicit_imports(Lux;
        skip=(LuxCore, LuxLib, LuxDeviceUtils, WeightInitializers, Base, Core, Lux)) ===
          nothing
    ## AbstractRNG seems to be a spurious detection in LuxFluxExt
    @test check_no_stale_explicit_imports(Lux;
        ignore=(:inputsize, :setup, :testmode, :trainmode, :update_state, :AbstractRNG)) ===
          nothing
end
