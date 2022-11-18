using Functors, Lux, Random, Test

rng = Random.default_rng()
Random.seed!(rng, 0)

@testset "AbstractExplicitLayer Interface" begin
    # Deprecated Functionality (Remove in v0.5)
    @test_deprecated Lux.initialparameters(rng, 10)
    @test_deprecated Lux.initialstates(rng, 10)
    @test_deprecated Lux.parameterlength(10)
    @test_deprecated Lux.statelength("testwarn")
end

@testset "update_state" begin
    st = (layer_1=(training=Val(true), val=1),
          layer_2=(layer_1=(val=2,), layer_2=(training=Val(true),)))

    # Deprecated Functionality (Remove in v0.5)
    @test_deprecated Lux.trainmode(st, true)
    @test_deprecated Lux.testmode(st, true)
end

@testset "Functors Compatibility" begin
    c = Parallel(+; chain=Chain(; dense_1=Dense(2 => 3), dense_2=Dense(3 => 5)),
                 dense_3=Dense(5 => 1))

    @test_nowarn fmap(println, c)

    l = Dense(2 => 2)
    new_model = fmap(x -> l, c)
    @test new_model.layers.chain.layers.dense_1 == l
    @test new_model.layers.chain.layers.dense_2 == l
    @test new_model.layers.dense_3 == l
end
