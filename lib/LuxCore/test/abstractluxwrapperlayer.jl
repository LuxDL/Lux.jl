using LuxCore, Test, Random

rng = LuxCore.Internal.default_rng()

include("common.jl")

@testset "AbstractLuxWrapperLayer Interface" begin
    model = ChainWrapper((; layer_1=Dense(5, 10), layer_2=Dense(10, 5)))
    x = randn(rng, Float32, 5)
    ps, st = LuxCore.setup(rng, model)

    @test fieldnames(typeof(ps)) == (:layer_1, :layer_2)
    @test fieldnames(typeof(st)) == (:layer_1, :layer_2)

    @test LuxCore.parameterlength(ps) ==
        LuxCore.parameterlength(model) ==
        LuxCore.parameterlength(model.layers.layer_1) +
        LuxCore.parameterlength(model.layers.layer_2)
    @test LuxCore.statelength(st) ==
        LuxCore.statelength(model) ==
        LuxCore.statelength(model.layers.layer_1) +
        LuxCore.statelength(model.layers.layer_2)

    @test LuxCore.apply(model, x, ps, st) == model(x, ps, st)

    @test LuxCore.stateless_apply(model, x, ps) == first(LuxCore.apply(model, x, ps, st))

    @test_nowarn println(model)
end
