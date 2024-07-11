@testitem "Aqua: Quality Assurance" tags=[:others] begin
    using Aqua, ChainRulesCore

    Aqua.test_all(Lux; piracies=false)
    Aqua.test_piracies(
        Lux; treat_as_own=[ChainRulesCore.frule, ChainRulesCore.rrule, Core.kwcall])
end

@testitem "Explicit Imports: Quality Assurance" setup=[SharedTestSetup] tags=[:others] begin
    # Load all trigger packages
    import Lux, ComponentArrays, ReverseDiff, Flux, SimpleChains, Tracker, Zygote, Enzyme

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

@testitem "doctests: Quality Assurance" tags=[:others] begin
    using Documenter

    doctestexpr = quote
        using SimpleChains: static
        using Flux: Flux
        using DynamicExpressions
        using Adapt, Lux, Random, Optimisers, Zygote
    end

    DocMeta.setdocmeta!(Lux, :DocTestSetup, doctestexpr; recursive=true)
    doctest(Lux; manual=false)
end
