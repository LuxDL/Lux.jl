using Aqua, ExplicitImports, MLDataDevices, Test

@testset "Aqua Tests" begin
    Aqua.test_all(MLDataDevices)
end

using FillArrays: FillArrays
using RecursiveArrayTools: RecursiveArrayTools
using SparseArrays: SparseArrays
using Zygote: Zygote

@testset "Explicit Imports" begin
    @test check_no_implicit_imports(MLDataDevices) === nothing
    @test check_no_stale_explicit_imports(MLDataDevices) === nothing
    @test check_no_self_qualified_accesses(MLDataDevices) === nothing
    @test check_all_explicit_imports_via_owners(MLDataDevices) === nothing
    @test check_all_qualified_accesses_via_owners(
        MLDataDevices; ignore=(:SparseArrays, :unsafe_free!)
    ) === nothing
    # mostly upstream problems
    @test_broken check_all_explicit_imports_are_public(MLDataDevices) === nothing
    @test_broken check_all_qualified_accesses_are_public(MLDataDevices) === nothing
end
