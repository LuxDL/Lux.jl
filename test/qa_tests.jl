@testitem "Aqua: Quality Assurance" tags=[:others] begin
    using Aqua, ChainRulesCore, ForwardDiff

    Aqua.test_all(Lux; ambiguities=false, piracies=false)
    Aqua.test_ambiguities(Lux;
        exclude=[ForwardDiff.jacobian, ForwardDiff.gradient,
            Lux.AutoDiffInternalImpl.batched_jacobian,
            Lux.AutoDiffInternalImpl.jacobian_vector_product,
            Lux.AutoDiffInternalImpl.jacobian_vector_product_impl])
    Aqua.test_piracies(Lux; treat_as_own=[Lux.outputsize])
end

@testitem "Explicit Imports: Quality Assurance" setup=[SharedTestSetup] tags=[:others] begin
    # Load all trigger packages
    import Lux, ComponentArrays, ReverseDiff, SimpleChains, Tracker, Zygote, Enzyme
    using ExplicitImports

    # Skip our own packages
    @test check_no_implicit_imports(
        Lux; skip=(Base, Core, LuxCore, MLDataDevices, LuxLib, WeightInitializers)) ===
          nothing
    @test check_no_stale_explicit_imports(
        Lux; ignore=(:setup, :testmode, :trainmode, :update_state)) === nothing
    @test check_no_self_qualified_accesses(Lux) === nothing
    @test check_all_explicit_imports_via_owners(Lux) === nothing
    @test check_all_qualified_accesses_via_owners(
        Lux; ignore=(:static_size, :_pullback, :AContext, :PtrArray)) === nothing
    @test_broken check_all_explicit_imports_are_public(Lux) === nothing  # mostly upstream problems
    @test_broken check_all_qualified_accesses_are_public(Lux) === nothing  # mostly upstream problems
end

# Some of the tests are flaky on prereleases
@testitem "doctests: Quality Assurance" tags=[:others] begin
    using Documenter

    doctestexpr = quote
        using SimpleChains: static
        using Adapt, Lux, Random, Optimisers, Zygote, NNlib
    end

    DocMeta.setdocmeta!(Lux, :DocTestSetup, doctestexpr; recursive=true)
    doctest(Lux; manual=false)
end
