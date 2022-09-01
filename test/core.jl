using Functors, Lux, Random, Test

rng = Random.default_rng()
Random.seed!(rng, 0)

# NOTE(@avik-pal): We have doctests for testing implementation of custom layers

@testset "AbstractExplicitLayer Interface" begin
    model = Dense(5, 6)
    ps, st = Lux.setup(rng, model)

    @test Lux.parameterlength(ps) == Lux.parameterlength(model)
    @test Lux.parameterlength(zeros(10, 2)) == 20
    @test Lux.statelength(st) == Lux.statelength(model)
    @test Lux.statelength(zeros(10, 2)) == 20
    @test Lux.statelength(Val(true)) == 1
    @test Lux.statelength((zeros(10), zeros(5, 2))) == 20
    @test Lux.statelength((layer_1=zeros(10), layer_2=zeros(5, 2))) == 20

    # Deprecated Functionality (Remove in v0.5)
    @test_deprecated Lux.initialparameters(rng, 10)
    @test_deprecated Lux.initialstates(rng, 10)
    @test_deprecated Lux.parameterlength(10)
    @test_deprecated Lux.statelength("testwarn")
end

@testset "update_state" begin
    st = (layer_1=(training=Val(true), val=1),
          layer_2=(layer_1=(val=2,), layer_2=(training=Val(true),)))

    st_ = Lux.testmode(st)

    @test st_.layer_1.training == Val(false) &&
          st_.layer_2.layer_2.training == Val(false) &&
          st_.layer_1.val == st.layer_1.val &&
          st_.layer_2.layer_1.val == st.layer_2.layer_1.val

    st = st_
    st_ = Lux.trainmode(st)

    @test st_.layer_1.training == Val(true) &&
          st_.layer_2.layer_2.training == Val(true) &&
          st_.layer_1.val == st.layer_1.val &&
          st_.layer_2.layer_1.val == st.layer_2.layer_1.val

    st_ = Lux.update_state(st, :val, -1)
    @test st_.layer_1.training == st.layer_1.training &&
          st_.layer_2.layer_2.training == st.layer_2.layer_2.training &&
          st_.layer_1.val == -1 &&
          st_.layer_2.layer_1.val == -1

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
