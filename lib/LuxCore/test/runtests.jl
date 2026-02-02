using LuxCore, Test, ParallelTestRunner

@testset "Extension Loading Checks (Fail)" begin
    @test !LuxCore.Internal.is_extension_loaded(Val(:Setfield))
    @test !LuxCore.Internal.is_extension_loaded(Val(:Functors))
    @test_throws ArgumentError LuxCore.Internal.setfield(1, 2, 3)
    @test_throws ArgumentError LuxCore.Internal.fmap(identity, 1)
    @test_throws ArgumentError LuxCore.Internal.fleaves(1)
end

using Functors, Setfield

@testset "Extension Loading Checks (Pass)" begin
    @test LuxCore.Internal.is_extension_loaded(Val(:Setfield))
    @test LuxCore.Internal.is_extension_loaded(Val(:Functors))
end

testsuite = find_tests(@__DIR__)
delete!(testsuite, "common")
runtests(LuxCore, ARGS; testsuite)
