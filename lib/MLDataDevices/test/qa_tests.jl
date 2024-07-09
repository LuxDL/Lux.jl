using Aqua, LuxDeviceUtils, Test

@testset "Aqua Tests" begin
    Aqua.test_all(LuxDeviceUtils)
end

import FillArrays, RecursiveArrayTools, SparseArrays, Zygote

@testset "Explicit Imports" begin
    @test check_no_implicit_imports(LuxDeviceUtils) === nothing
    @test check_no_stale_explicit_imports(LuxDeviceUtils) === nothing
    @test check_no_self_qualified_accesses(LuxDeviceUtils) === nothing
    @test check_all_explicit_imports_via_owners(LuxDeviceUtils) === nothing
    @test check_all_qualified_accesses_via_owners(LuxDeviceUtils) === nothing
    @test_broken check_all_explicit_imports_are_public(LuxDeviceUtils) === nothing  # mostly upstream problems
    @test_broken check_all_qualified_accesses_are_public(LuxDeviceUtils) === nothing  # mostly upstream problem
end
