@testitem "All Parameter Freezing" setup = [SharedTestSetup] tags = [:misc] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset "NamedTuple" begin
            d = Dense(5 => 5)
            psd, std = dev.(Lux.setup(rng, d))

            fd, ps, st = Lux.Experimental.freeze(d, psd, std, nothing)
            @test length(keys(ps)) == 0
            @test length(keys(st)) == 2
            @test sort([keys(st)...]) == [:frozen_params, :states]
            @test sort([keys(st.frozen_params)...]) == [:bias, :weight]

            x = aType(randn(rng, Float32, 5, 1))

            @test d(x, psd, std)[1] == fd(x, ps, st)[1]
            @jet fd(x, ps, st)
            @test_gradients(sumabs2first, fd, x, ps, st; atol=1.0f-3, rtol=1.0f-3)
        end

        @testset "ComponentArray" begin
            m = Chain(Lux.Experimental.freeze(Dense(1 => 3, tanh)), Dense(3 => 1))
            ps, st = Lux.setup(rng, m)
            st = dev(st)
            ps_c = dev(ComponentVector(ps))
            ps = dev(ps)
            x = aType(randn(rng, Float32, 1, 2))

            @test m(x, ps, st)[1] == m(x, ps_c, st)[1]
            @jet m(x, ps_c, st)
            @test_gradients(sumabs2first, m, x, ps_c, st; atol=1.0f-3, rtol=1.0f-3,)
        end

        @testset "LuxDL/Lux.jl#427" begin
            m = Dense(1 => 1)
            ps, st = Lux.setup(rng, m)
            st = dev(st)
            ps_c = dev(ComponentVector(ps))
            ps = dev(ps)

            fd, psf, stf = Lux.Experimental.freeze(m, ps, st)

            @test fd isa Lux.Experimental.FrozenLayer
            @test psf isa NamedTuple{}
            @test sort([keys(stf)...]) == [:frozen_params, :states]
            @test sort([keys(stf.frozen_params)...]) == [:bias, :weight]

            fd, psf, stf = Lux.Experimental.freeze(m, ps_c, st)

            @test fd isa Lux.Experimental.FrozenLayer
            @test psf isa NamedTuple{}
            @test sort([keys(stf)...]) == [:frozen_params, :states]
            @test sort([keys(stf.frozen_params)...]) == [:bias, :weight]
        end
    end
end

@testitem "Partial Freezing" setup = [SharedTestSetup] tags = [:misc] begin
    using Lux.Experimental: FrozenLayer

    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        d = Dense(5 => 5)
        psd, std = dev.(Lux.setup(rng, d))

        fd, ps, st = Lux.Experimental.freeze(d, psd, std, (:weight,))
        @test length(keys(ps)) == 1
        @test length(keys(st)) == 2
        @test sort([keys(st)...]) == [:frozen_params, :states]
        @test sort([keys(st.frozen_params)...]) == [:weight]
        @test sort([keys(ps)...]) == [:bias]

        x = aType(randn(rng, Float32, 5, 1))

        @test d(x, psd, std)[1] == fd(x, ps, st)[1]
        @jet fd(x, ps, st)
        @test_gradients(sumabs2first, fd, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

        fd = Lux.Experimental.freeze(d, ())
        @test fd === d

        fd = Lux.Experimental.freeze(d, nothing)
        display(fd)
        @test fd isa Lux.Experimental.FrozenLayer

        und = Lux.Experimental.unfreeze(fd)
        @test und === d

        und, psd, std = Lux.Experimental.unfreeze(fd, ps, st)
        @test und === d
        @test und(x, psd, std)[1] == fd(x, ps, st)[1]
    end
end

@testitem "#1319: Freezing a @compact layer" setup = [SharedTestSetup] tags = [:misc] begin
    using Lux, Random, Functors

    function freeze_by_name(d, ps, st, name::KeyPath)
        if name == KeyPath(:layer_1)
            return Lux.Experimental.freeze(d, ps, st, (:weight, :bias))
        end
        return d, ps, st
    end

    model = @compact(
        layer_1 = Dense(2 => 5),
        layer_2 = Dense(5 => 5),
        layer_3 = Dense(5 => 1),
        data = rand(Float32, 2, 5),
        data_nt = @non_trainable(rand(Float32, 1, 4))
    ) do x
        x = layer_1(x)
        x = layer_2(x)
        x = layer_3(x)
        @return x
    end
    ps, st = Lux.setup(Random.default_rng(), model)

    model_frozen, ps_frozen, st_frozen = Lux.Experimental.layer_map(
        freeze_by_name, model, ps, st
    )
    @test model_frozen.layers.layer_1 isa Lux.Experimental.FrozenLayer
    @test !isempty(ps.layer_1)
    @test isempty(ps_frozen.layer_1)
    @test isempty(st.layer_1)
    @test !isempty(st_frozen.layer_1)
end
