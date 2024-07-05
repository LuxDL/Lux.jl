@testitem "Aqua: Quality Assurance" tags=[:others] begin
    using Aqua

    Aqua.test_all(LuxLib; unbound_args=(; broken=true))  # GPUArraysCore.AnyGPUArray causes problem here
end

@testitem "Explicit Imports" tags=[:others] begin
    import cuDNN, CUDA, ForwardDiff, ReverseDiff, Tracker, AMDGPU, NNlib

    using ExplicitImports

    @test check_no_implicit_imports(LuxLib) === nothing
    @test check_no_stale_explicit_imports(LuxLib, ignore=(:TrackedVector,)) === nothing
    @test check_no_self_qualified_accesses(LuxLib) === nothing
    @test check_all_explicit_imports_via_owners(LuxLib) === nothing
    @test check_all_qualified_accesses_via_owners(LuxLib) === nothing
    @test_broken check_all_explicit_imports_are_public(LuxLib) === nothing  # mostly upstream problems
    @test_broken check_all_qualified_accesses_are_public(LuxLib) === nothing  # mostly upstream problems
end
