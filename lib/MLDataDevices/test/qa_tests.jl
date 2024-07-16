using Aqua, ExplicitImports, DeviceUtils, Test

@testset "Aqua Tests" begin
    Aqua.test_all(DeviceUtils)
end

import FillArrays, RecursiveArrayTools, SparseArrays, Zygote

@testset "Explicit Imports" begin
    @test check_no_implicit_imports(DeviceUtils) === nothing
    @test check_no_stale_explicit_imports(DeviceUtils) === nothing
    @test check_no_self_qualified_accesses(DeviceUtils) === nothing
    @test check_all_explicit_imports_via_owners(DeviceUtils) === nothing
    @test check_all_qualified_accesses_via_owners(DeviceUtils) === nothing
    @test_broken check_all_explicit_imports_are_public(DeviceUtils) === nothing  # mostly upstream problems
    @test_broken check_all_qualified_accesses_are_public(DeviceUtils) === nothing  # mostly upstream problem
end
