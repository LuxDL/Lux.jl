@testitem "Aqua: Quality Assurance" tags=[:others] begin
    using Aqua, ChainRulesCore

    Aqua.test_all(LuxLib; ambiguities=false, piracies=false)
    Aqua.test_ambiguities(LuxLib; recursive=false,
        exclude=[conv, ∇conv_data, ∇conv_filter, depthwiseconv, ChainRulesCore.frule])
    Aqua.test_piracies(LuxLib; treat_as_own=[conv, ∇conv_data, ∇conv_filter, depthwiseconv])
end

@testitem "Explicit Imports" tags=[:others] begin
    import ReverseDiff, Tracker, NNlib
    using ExplicitImports

    @test check_no_implicit_imports(LuxLib) === nothing
    @test check_no_stale_explicit_imports(LuxLib; ignore=(:TrackedVector,)) === nothing
    @test check_no_self_qualified_accesses(LuxLib) === nothing
    @test check_all_explicit_imports_via_owners(LuxLib) === nothing
    @test check_all_qualified_accesses_via_owners(LuxLib) === nothing
    @test_broken check_all_explicit_imports_are_public(LuxLib) === nothing  # mostly upstream problems
    @test_broken check_all_qualified_accesses_are_public(LuxLib) === nothing  # mostly upstream problems
end
