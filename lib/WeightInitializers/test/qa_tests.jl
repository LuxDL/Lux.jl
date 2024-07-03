@testitem "Aqua: Quality Assurance" begin
    using Aqua

    Aqua.test_all(WeightInitializers; ambiguities=false)
    Aqua.test_ambiguities(WeightInitializers; recursive=false)
end

@testitem "Explicit Imports: Quality Assurance" setup=[SharedTestSetup] begin
    using ExplicitImports

    @test check_no_implicit_imports(WeightInitializers) === nothing
    @test check_no_stale_explicit_imports(WeightInitializers) === nothing
    @test check_no_self_qualified_accesses(WeightInitializers) === nothing
end

@testitem "doctests: Quality Assurance" begin
    using Documenter

    doctestexpr = :(using Random, WeightInitializers)

    DocMeta.setdocmeta!(WeightInitializers, :DocTestSetup, doctestexpr; recursive=true)
    doctest(WeightInitializers; manual=false)
end
