using LuxCore, Test, ExplicitImports, Aqua

@testset "Quality Assurance" begin
    Aqua.test_all(LuxCore)
    ExplicitImports.test_explicit_imports(LuxCore)
end
