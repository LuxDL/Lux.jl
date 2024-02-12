@testitem "All Parameter Freezing" setup=[SharedTestSetup] begin
    rng = get_stable_rng(12345)

    @testset "$mode" for (mode, aType, device, ongpu) in MODES
        @testset "NamedTuple" begin
            d = Dense(5 => 5)
            psd, std = Lux.setup(rng, d) .|> device

            fd, ps, st = Lux.Experimental.freeze(d, psd, std, nothing)
            @test length(keys(ps)) == 0
            @test length(keys(st)) == 2
            @test sort([keys(st)...]) == [:frozen_params, :states]
            @test sort([keys(st.frozen_params)...]) == [:bias, :weight]

            x = randn(rng, Float32, 5, 1) |> aType

            @test d(x, psd, std)[1] == fd(x, ps, st)[1]

            @jet fd(x, ps, st)
            __f = (x, ps) -> sum(first(fd(x, ps, st)))

            @eval @test_gradients $__f $x $ps atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu
        end

        @testset "ComponentArray" begin
            m = Chain(Lux.Experimental.freeze(Dense(1 => 3, tanh)), Dense(3 => 1))
            ps, st = Lux.setup(rng, m)
            st = st |> device
            ps_c = ComponentVector(ps) |> device
            ps = ps |> device
            x = randn(rng, Float32, 1, 2) |> aType

            @test m(x, ps, st)[1] == m(x, ps_c, st)[1]

            @jet m(x, ps_c, st)
            __f = (x, ps) -> sum(first(m(x, ps, st)))
            @eval @test_gradients $__f $x $ps_c atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu
        end

        @testset "LuxDL/Lux.jl#427" begin
            m = Dense(1 => 1)
            ps, st = Lux.setup(rng, m)
            st = st |> device
            ps_c = ComponentVector(ps) |> device
            ps = ps |> device

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

@testitem "Partial Freezing" setup=[SharedTestSetup] begin
    rng = get_stable_rng(12345)

    @testset "$mode" for (mode, aType, device, ongpu) in MODES
        d = Dense(5 => 5)
        psd, std = Lux.setup(rng, d) .|> device

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
        @eval @test_gradients $__f $x $ps atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu
    end
end
