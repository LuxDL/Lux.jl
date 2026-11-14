using LuxCore, Test, ParallelTestRunner

@testset "Extension Loading Checks (Fail)" begin
    @test !LuxCore.Internal.is_extension_loaded(Val(:Setfield))
    @test !LuxCore.Internal.is_extension_loaded(Val(:Functors))
    @test_throws ArgumentError LuxCore.Internal.setfield(1, 2, 3)
    @test_throws ArgumentError LuxCore.Internal.fmap(identity, 1)
    @test_throws ArgumentError LuxCore.Internal.fleaves(1)
end

using Functors

include("common.jl")

@testset "Functors without Setfield" begin
    @test !LuxCore.Internal.is_extension_loaded(Val(:Setfield))
    @test LuxCore.Internal.is_extension_loaded(Val(:Functors))

    # AbstractLuxContainerLayer
    c1 = Chain((; layer_1=Dense(5, 10), layer_2=Dense(10, 5)))
    c1_mapped = Functors.fmap(l -> l isa Dense ? Dense(l.in * 2, l.out * 2) : l, c1)
    @test c1_mapped isa Chain
    @test c1_mapped.layers.layer_1.in == 10
    @test c1_mapped.layers.layer_1.out == 20
    @test c1_mapped.layers.layer_2.in == 20
    @test c1_mapped.layers.layer_2.out == 10

    c2 = Chain2(Dense(5, 10), Dense(10, 5))
    c2_mapped = Functors.fmap(l -> l isa Dense ? Dense(l.in * 2, l.out * 2) : l, c2)
    @test c2_mapped isa Chain2
    @test c2_mapped.layer1.in == 10
    @test c2_mapped.layer1.out == 20
    @test c2_mapped.layer2.in == 20
    @test c2_mapped.layer2.out == 10

    # AbstractLuxWrapperLayer
    cw = ChainWrapper((; layer_1=Dense(5, 10), layer_2=Dense(10, 5)))
    cw_mapped = Functors.fmap(l -> l isa Dense ? Dense(l.in * 2, l.out * 2) : l, cw)
    @test cw_mapped isa ChainWrapper
    @test cw_mapped.layers.layer_1.in == 10
    @test cw_mapped.layers.layer_1.out == 20
    @test cw_mapped.layers.layer_2.in == 20
    @test cw_mapped.layers.layer_2.out == 10

    dw = DenseWrapper(Dense(5, 10))
    dw_mapped = Functors.fmap(l -> l isa Dense ? Dense(l.in * 2, l.out * 2) : l, dw)
    @test dw_mapped isa DenseWrapper
    @test dw_mapped.layer.in == 10
    @test dw_mapped.layer.out == 20

    # LuxCore.Internal.setfield without Setfield
    c_updated = LuxCore.Internal.setfield(
        c1, :layers, (; layer_1=Dense(1, 2), layer_2=Dense(3, 4))
    )
    @test c_updated.layers.layer_1.in == 1
    c_updated2 = LuxCore.Internal.setfield(
        c1, (:layers, (; layer_1=Dense(2, 3), layer_2=Dense(4, 5)))
    )
    @test c_updated2.layers.layer_1.in == 2
    @test_throws ArgumentError LuxCore.Internal.setfield(c1, :nonexistent, 1)
    @test_throws ArgumentError LuxCore.Internal.setfield(1, 2, 3)
end

using Setfield

@testset "Extension Loading Checks (Pass)" begin
    @test LuxCore.Internal.is_extension_loaded(Val(:Setfield))
    @test LuxCore.Internal.is_extension_loaded(Val(:Functors))
end

testsuite = find_tests(@__DIR__)
delete!(testsuite, "common")
runtests(LuxCore, ARGS; testsuite)
