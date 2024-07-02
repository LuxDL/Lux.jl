@testitem "All Parameter Freezing" setup=[SharedTestSetup] tags=[:contrib] begin
    Enzyme.API.runtimeActivity!(true)
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset "NamedTuple" begin
            d = Dense(5 => 5)
            psd, std = Lux.setup(rng, d) .|> dev

            fd, ps, st = Lux.Experimental.freeze(d, psd, std, nothing)
            @test length(keys(ps)) == 0
            @test length(keys(st)) == 2
            @test sort([keys(st)...]) == [:frozen_params, :states]
            @test sort([keys(st.frozen_params)...]) == [:bias, :weight]

            x = randn(rng, Float32, 5, 1) |> aType

            @test d(x, psd, std)[1] == fd(x, ps, st)[1]

            @jet fd(x, ps, st)

            __f = x -> sum(first(fd(x, ps, st)))
            test_gradients(__f, x; atol=1.0f-3, rtol=1.0f-3)
        end

        @testset "ComponentArray" begin
            m = Chain(Lux.Experimental.freeze(Dense(1 => 3, tanh)), Dense(3 => 1))
            ps, st = Lux.setup(rng, m)
            st = st |> dev
            ps_c = ComponentVector(ps) |> dev
            ps = ps |> dev
            x = randn(rng, Float32, 1, 2) |> aType

            @test m(x, ps, st)[1] == m(x, ps_c, st)[1]

            @jet m(x, ps_c, st)

            __f = (x, ps) -> sum(first(m(x, ps, st)))
            test_gradients(__f, x, ps_c; atol=1.0f-3, rtol=1.0f-3)
        end

        @testset "LuxDL/Lux.jl#427" begin
            m = Dense(1 => 1)
            ps, st = Lux.setup(rng, m)
            st = st |> dev
            ps_c = ComponentVector(ps) |> dev
            ps = ps |> dev

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

@testitem "Partial Freezing" setup=[SharedTestSetup] tags=[:contrib] begin
    using Lux.Experimental: FrozenLayer

    Enzyme.API.runtimeActivity!(true)
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        d = Dense(5 => 5)
        psd, std = Lux.setup(rng, d) .|> dev

        fd, ps, st = Lux.Experimental.freeze(d, psd, std, (:weight,))
        @test length(keys(ps)) == 1
        @test length(keys(st)) == 2
        @test sort([keys(st)...]) == [:frozen_params, :states]
        @test sort([keys(st.frozen_params)...]) == [:weight]
        @test sort([keys(ps)...]) == [:bias]

        x = randn(rng, Float32, 5, 1) |> aType

        @test d(x, psd, std)[1] == fd(x, ps, st)[1]

        @jet fd(x, ps, st)
        __f = (x, ps) -> sum(first(fd(x, ps, st)))
        test_gradients(__f, x, ps; atol=1.0f-3, rtol=1.0f-3)

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
