@testitem "Layer Map" setup=[SharedTestSetup] tags=[:contrib] begin
    using Setfield, Functors

    function occurs_in(kp::KeyPath, x::KeyPath)
        length(kp)≤length(x)&&return all(==(x[i], kp[i]) for i in 1:length(kp))
        return false
    end

    function zero_dense_params_1(l, ps, st, name)
        if l isa Dense&&occurs_in(KeyPath(:chain), name)
            @set! ps.weight = zero(ps.weight)
            @set! ps.bias = zero(ps.bias)
        end
        return l, ps, st
    end

    function zero_dense_params_2(l, ps, st, name)
        if l isa Dense
            @set! ps.weight = zero(ps.weight)
            @set! ps.bias = zero(ps.bias)
        end
        return l, ps, st
    end

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        c = Parallel(
            +; chain=Chain(; dense_1=Dense(2 => 3), bn=BatchNorm(3), dense_2=Dense(3 => 5)),
            dense_3=Dense(5 => 1))

        rng = StableRNG(12345)
        ps, st = Lux.setup(rng, c) |> dev

        c_, ps_, st_ = Lux.Experimental.layer_map(zero_dense_params_1, c, ps, st)

        @test all(iszero, ps_.chain.dense_1.weight)
        @test all(iszero, ps_.chain.dense_1.bias)
        @test all(iszero, ps_.chain.dense_2.weight)
        @test all(iszero, ps_.chain.dense_2.bias)
        @test !all(iszero, ps_.dense_3.weight)
        @test !all(iszero, ps_.dense_3.bias)

        # Custom Layers -- See https://github.com/LuxDL/Lux.jl/issues/187
        struct SimpleCustom{L1, L2} <: Lux.AbstractLuxContainerLayer{(:dense, :conv)}
            dense::L1
            conv::L2
        end

        l = SimpleCustom(Dense(3 => 2), Conv((3,), 3 => 2))

        ps, st = Lux.setup(rng, l) |> dev

        l_, ps_, st_ = Lux.Experimental.layer_map(zero_dense_params_2, l, ps, st)

        @test all(iszero, ps_.dense.weight)
        @test all(iszero, ps_.dense.bias)
        @test !all(iszero, ps_.conv.weight)
        @test !all(iszero, ps_.conv.bias)

        # Custom Wrapper
        struct SimpleWrapper{L} <: Lux.AbstractLuxWrapperLayer{:model}
            model::L
        end

        l = SimpleWrapper(SimpleCustom(Dense(3 => 2), Conv((3,), 3 => 2)))

        ps, st = Lux.setup(rng, l) |> dev

        l_, ps_, st_ = Lux.Experimental.layer_map(zero_dense_params_2, l, ps, st)

        @test all(iszero, ps_.dense.weight)
        @test all(iszero, ps_.dense.bias)
        @test !all(iszero, ps_.conv.weight)
        @test !all(iszero, ps_.conv.bias)
    end
end
