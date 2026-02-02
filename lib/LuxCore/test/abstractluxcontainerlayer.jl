using LuxCore, Test, Random

rng = LuxCore.Internal.default_rng()

include("common.jl")

@testset "AbstractLuxContainerLayer Interface" begin
    model = Chain((; layer_1=Dense(5, 5), layer_2=Dense(5, 6)))
    x = randn(rng, Float32, 5)
    ps, st = LuxCore.setup(rng, model)

    @test fieldnames(typeof(ps)) == (:layers,)
    @test fieldnames(typeof(st)) == (:layers,)

    @test LuxCore.parameterlength(ps) ==
        LuxCore.parameterlength(model) ==
        LuxCore.parameterlength(model.layers[1]) + LuxCore.parameterlength(model.layers[2])
    @test LuxCore.statelength(st) ==
        LuxCore.statelength(model) ==
        LuxCore.statelength(model.layers[1]) + LuxCore.statelength(model.layers[2])

    @test LuxCore.apply(model, x, ps, st) == model(x, ps, st)

    @test LuxCore.stateless_apply(model, x, ps) == first(LuxCore.apply(model, x, ps, st))

    @test_nowarn println(model)

    model = Chain2(Dense(5, 5), Dense(5, 6))
    x = randn(rng, Float32, 5)
    ps, st = LuxCore.setup(rng, model)

    @test LuxCore.parameterlength(ps) ==
        LuxCore.parameterlength(model) ==
        LuxCore.parameterlength(model.layer1) + LuxCore.parameterlength(model.layer2)
    @test LuxCore.statelength(st) ==
        LuxCore.statelength(model) ==
        LuxCore.statelength(model.layer1) + LuxCore.statelength(model.layer2)

    @test LuxCore.apply(model, x, ps, st) == model(x, ps, st)

    @test LuxCore.stateless_apply(model, x, ps) == first(LuxCore.apply(model, x, ps, st))

    @test_nowarn println(model)
end
