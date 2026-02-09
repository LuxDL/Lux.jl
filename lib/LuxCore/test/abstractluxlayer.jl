using LuxCore, Test, Random, Functors

rng = LuxCore.Internal.default_rng()

include("common.jl")

@testset "AbstractLuxLayer Interface" begin
    @testset "Custom Layer" begin
        model = Dense(5, 6)
        x = randn(rng, Float32, 5)
        ps, st = LuxCore.setup(rng, model)

        @test LuxCore.parameterlength(ps) == LuxCore.parameterlength(model)
        @test LuxCore.statelength(st) == LuxCore.statelength(model)

        @test LuxCore.apply(model, x, ps, st) == model(x, ps, st)

        @test LuxCore.stateless_apply(model, x, ps) ==
            first(LuxCore.apply(model, x, ps, NamedTuple()))

        @test_nowarn println(model)

        @testset for wrapper in (DenseWrapper, DenseWrapper2)
            model2 = wrapper(model)
            ps, st = LuxCore.setup(rng, model2)

            @test LuxCore.parameterlength(ps) == LuxCore.parameterlength(model2)
            @test LuxCore.statelength(st) == LuxCore.statelength(model2)

            @test model2(x, ps, st)[1] == model(x, ps, st)[1]

            @test_nowarn println(model2)
        end
    end

    @testset "Default Fallbacks" begin
        struct NoParamStateLayer <: AbstractLuxLayer end

        layer = NoParamStateLayer()
        @test LuxCore.initialparameters(rng, layer) == NamedTuple()
        @test LuxCore.initialstates(rng, layer) == NamedTuple()

        @test LuxCore.parameterlength(zeros(10, 2)) == 20
        @test LuxCore.statelength(zeros(10, 2)) == 20
        @test LuxCore.statelength(Val(true)) == 1
        @test LuxCore.statelength((zeros(10), zeros(5, 2))) == 20
        @test LuxCore.statelength((layer_1=zeros(10), layer_2=zeros(5, 2))) == 20

        @test LuxCore.initialparameters(rng, NamedTuple()) == NamedTuple()
        @test_throws MethodError LuxCore.initialparameters(rng, ())
        @test LuxCore.initialparameters(rng, nothing) == NamedTuple()
        @test LuxCore.initialparameters(rng, (nothing, layer)) ==
            (NamedTuple(), NamedTuple())

        @test LuxCore.initialstates(rng, NamedTuple()) == NamedTuple()
        @test_throws MethodError LuxCore.initialstates(rng, ())
        @test LuxCore.initialstates(rng, nothing) == NamedTuple()
        @test LuxCore.initialparameters(rng, (nothing, layer)) ==
            (NamedTuple(), NamedTuple())
    end
end
