@testitem "Aqua: Quality Assurance" tags=[:nworkers] begin
    using Aqua
    Aqua.test_all(LuxLib)
end

@testitem "Explicit Imports" tags=[:nworkers] begin
    import cuDNN, CUDA, ForwardDiff, ReverseDiff, Tracker, AMDGPU, NNlib

    using ExplicitImports

    # Skip our own packages
    @test check_no_implicit_imports(LuxLib; skip=(NNlib, Base, Core)) === nothing
    @test check_no_stale_explicit_imports(LuxLib; ignore=(:TrackedVector,)) === nothing
end
