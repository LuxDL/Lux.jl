@testitem "Aqua: Quality Assurance" tags=[:misc] begin
    using Aqua, ChainRulesCore, EnzymeCore, NNlib
    using EnzymeCore: EnzymeRules

    Aqua.test_all(
        LuxLib; ambiguities=false, piracies=false, stale_deps=Sys.ARCH === :x86_64)
    Aqua.test_ambiguities(LuxLib; recursive=false,
        exclude=[conv, ∇conv_data, ∇conv_filter, depthwiseconv, ChainRulesCore.frule])
    Aqua.test_piracies(LuxLib;
        treat_as_own=[conv, ∇conv_data, ∇conv_filter, depthwiseconv,
            EnzymeRules.augmented_primal, EnzymeRules.reverse])
end

@testitem "Explicit Imports" tags=[:misc] setup=[SharedTestSetup] begin
    using ExplicitImports

    @test check_no_implicit_imports(LuxLib) === nothing
    @test check_no_stale_explicit_imports(
        LuxLib; ignore=(:TrackedVector, :batched_mul, :batched_matmul)) === nothing
    @test check_no_self_qualified_accesses(LuxLib) === nothing
    @test check_all_explicit_imports_via_owners(LuxLib) === nothing
    @test check_all_qualified_accesses_via_owners(LuxLib) === nothing
    @test_broken check_all_explicit_imports_are_public(LuxLib) === nothing  # mostly upstream problems
    @test_broken check_all_qualified_accesses_are_public(LuxLib) === nothing  # mostly upstream problems
end
