@testitem "Aqua: Quality Assurance" begin
    using Aqua

    Aqua.test_all(WeightInitializers; ambiguities=false)
    Aqua.test_ambiguities(WeightInitializers; recursive=false)
end

@testitem "Explicit Imports: Quality Assurance" setup = [SharedTestSetup] begin
    using ExplicitImports

    @test check_no_implicit_imports(WeightInitializers) === nothing
    @test check_no_stale_explicit_imports(WeightInitializers; ignore=(:randn!, :rand!)) ===
        nothing
    @test check_no_self_qualified_accesses(WeightInitializers) === nothing
    @test check_all_explicit_imports_via_owners(WeightInitializers) === nothing
    @test check_all_qualified_accesses_via_owners(WeightInitializers) === nothing
    @test_broken check_all_explicit_imports_are_public(WeightInitializers) === nothing  # mostly upstream problems

    try  # FIXME: Soft fail for now
        acc = check_all_qualified_accesses_are_public(WeightInitializers)
        @test acc === nothing
    catch
        @test_broken check_all_qualified_accesses_are_public(WeightInitializers) === nothing
    end
end

@testitem "doctests: Quality Assurance" begin
    using Documenter

    doctestexpr = :(using Random, WeightInitializers)

    DocMeta.setdocmeta!(WeightInitializers, :DocTestSetup, doctestexpr; recursive=true)
    doctest(WeightInitializers; manual=false)
end
