using Lux, ComponentArrays, CUDA, Functors, ReverseDiff, Random, Optimisers, Zygote, Test

include("test_utils.jl")

rng = Random.default_rng()
Random.seed!(rng, 0)

@testset "_nfan" begin
    # Fallback
    @test Lux._nfan() == (1, 1)
    # Vector
    @test Lux._nfan(4) == (1, 4)
    # Matrix
    @test Lux._nfan(4, 5) == (5, 4)
    # Tuple
    @test Lux._nfan((4, 5, 6)) == Lux._nfan(4, 5, 6)
    # Convolution
    @test Lux._nfan(4, 5, 6) == 4 .* (5, 6)
end

@testset "replicate" begin
    @test randn(rng, 10, 2) != randn(rng, 10, 2)
    @test randn(Lux.replicate(rng), 10, 2) == randn(Lux.replicate(rng), 10, 2)

    if CUDA.functional()
        curng = CUDA.RNG()
        @test randn(curng, 10, 2) != randn(curng, 10, 2)
        @test randn(Lux.replicate(curng), 10, 2) == randn(Lux.replicate(curng), 10, 2)
    end
end

@testset "istraining" begin
    @test Lux.istraining(Val(true))
    @test !Lux.istraining(Val(false))
    @test !Lux.istraining((training=Val(false),))
    @test Lux.istraining((training=Val(true),))
    @test !Lux.istraining((no_training=1,))
    @test_throws MethodError Lux.istraining((training=true,))
end

@testset "multigate" begin
    x = randn(rng, 10, 1)
    x1, x2 = Lux.multigate(x, Val(2))

    @test x1 == x[1:5, :]
    @test x2 == x[6:10, :]
    @inferred Lux.multigate(x, Val(2))

    run_JET_tests(Lux.multigate, x, Val(2))

    x = randn(rng, 10)
    x1, x2 = Lux.multigate(x, Val(2))

    @test x1 == x[1:5]
    @test x2 == x[6:10]
    @inferred Lux.multigate(x, Val(2))

    run_JET_tests(Lux.multigate, x, Val(2))
end

@testset "ComponentArrays" begin
    ps = (weight=randn(rng, 3, 4), bias=randn(rng, 4))
    p_flat, re = Optimisers.destructure(ps)
    ps_c = ComponentArray(ps)

    @test ps_c.weight == ps.weight
    @test ps_c.bias == ps.bias

    @test p_flat == getdata(ps_c)
    @test -p_flat == getdata(-ps_c)
    @test zero(p_flat) == getdata(zero(ps_c))

    @test_nowarn similar(ps_c, 10)
    @test_nowarn similar(ps_c)

    ps_c_f, ps_c_re = Functors.functor(ps_c)
    @test ps_c_f == ps
    @test ps_c_re(ps_c_f) == ps_c

    # Empty ComponentArray test
    @test_nowarn display(ComponentArray(NamedTuple()))
    println()

    # Optimisers
    opt = Optimisers.ADAM(0.001f0)
    st_opt = Optimisers.setup(opt, ps_c)

    @test_nowarn Optimisers.update(st_opt, ps_c, ps_c)
    @test_nowarn Optimisers.update!(st_opt, ps_c, ps_c)

    if CUDA.functional()
        ps_c = ps_c |> gpu
        st_opt = Optimisers.setup(opt, ps_c)

        @test_nowarn Optimisers.update(st_opt, ps_c, ps_c)
        @test_nowarn Optimisers.update!(st_opt, ps_c, ps_c)
    end
end

@testset "_init_hidden_state" begin
    rnn = RNNCell(3 => 5; init_state=Lux.zeros32)
    x = randn(rng, Float32, 3, 2, 2)
    @test Lux._init_hidden_state(rng, rnn, view(x, :, 1, :)) == zeros(Float32, 5, 2)

    if CUDA.functional()
        x = x |> gpu
        @test Lux._init_hidden_state(rng, rnn, view(x, :, 1, :)) ==
              CUDA.zeros(Float32, 5, 2)
    end
end
