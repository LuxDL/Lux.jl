@testitem "replicate" setup=[SharedTestSetup] tags=[:others] begin
    @testset "$mode" for (mode, aType, device, ongpu) in MODES
        _rng = get_default_rng(mode)

        if mode == "AMDGPU"
            @test_skip randn(_rng, 10, 2) != randn(_rng, 10, 2)
            @test_skip randn(Lux.replicate(_rng), 10, 2) ==
                       randn(Lux.replicate(_rng), 10, 2)
        else
            @test randn(_rng, 10, 2) != randn(_rng, 10, 2)
            @test randn(Lux.replicate(_rng), 10, 2) == randn(Lux.replicate(_rng), 10, 2)
        end
    end
end

@testitem "istraining" tags=[:others] begin
    @test Lux.istraining(Val(true))
    @test !Lux.istraining(Val(false))
    @test !Lux.istraining((training=Val(false),))
    @test Lux.istraining((training=Val(true),))
    @test !Lux.istraining((no_training=1,))
    @test Lux.istraining((training=true,))
    @test !Lux.istraining((training=false,))
end

@testitem "multigate" setup=[SharedTestSetup] tags=[:others] begin
    rng = get_stable_rng(12345)

    @testset "$mode" for (mode, aType, device, ongpu) in MODES
        x = randn(rng, 10, 1) |> aType
        x1, x2 = Lux.multigate(x, Val(2))

        @test x1 == x[1:5, :]
        @test x2 == x[6:10, :]

        @jet Lux.multigate(x, Val(2))

        x = randn(rng, 10) |> aType
        x1, x2 = Lux.multigate(x, Val(2))

        @test x1 == x[1:5]
        @test x2 == x[6:10]

        @jet Lux.multigate(x, Val(2))

        x = rand(6, 5) |> aType
        __f = x -> begin
            x1, x2, x3 = Lux.multigate(x, Val(3))
            return sum(x1) + sum(x3 .+ x2 .^ 2)
        end
        res, (dx,) = Zygote.withgradient(__f, x)

        @jet Lux.multigate(x, Val(3))

        @test res ≈ sum(x[1:2, :]) + sum(x[5:6, :]) + sum(abs2, x[3:4, :])
        @test dx ≈ aType([ones(2, 5); Array(x[3:4, :] .* 2); ones(2, 5)])

        @eval @test_gradients $__f $x atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu
    end
end

@testitem "ComponentArrays" setup=[SharedTestSetup] tags=[:others] begin
    using Optimisers, Functors

    rng = get_stable_rng(12345)

    @testset "$mode" for (mode, aType, device, ongpu) in MODES
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
        @test_nowarn __display(ComponentArray(NamedTuple()))
        println()

        # Optimisers
        opt = Adam(0.001f0)
        ps_c = ps_c |> device
        st_opt = Optimisers.setup(opt, ps_c)

        @test_nowarn Optimisers.update(st_opt, ps_c, ps_c)
        @test_nowarn Optimisers.update!(st_opt, ps_c, ps_c)
    end

    # Ref: https://github.com/LuxDL/Lux.jl/issues/243
    nn = Chain(Dense(4, 3), Dense(3, 2))
    ps, st = Lux.setup(rng, nn)

    l2reg(p) = sum(abs2, ComponentArray(p))
    @test_nowarn Zygote.gradient(l2reg, ps)
end

@testitem "_init_hidden_state" setup=[SharedTestSetup] tags=[:recurrent_layers] begin
    rng = get_stable_rng(12345)

    @testset "$mode" for (mode, aType, device, ongpu) in MODES
        rnn = RNNCell(3 => 5; init_state=Lux.zeros32)
        x = randn(rng, Float32, 3, 2, 2)
        @test Lux._init_hidden_state(rng, rnn, view(device(x), :, 1, :)) ==
              aType(zeros(Float32, 5, 2))
    end
end

@testitem "FP Conversions" setup=[SharedTestSetup] tags=[:others] begin
    rng = get_stable_rng(12345)

    @testset "$mode" for (mode, aType, device, ongpu) in MODES
        model = Chain(Dense(1 => 16, relu), Chain(Dense(16 => 1), Dense(1 => 1)),
            BatchNorm(1); disable_optimizations=true)

        for (f, ftype) in zip((f16, f32, f64), (Float16, Float32, Float64))
            ps, st = Lux.setup(rng, model) |> device |> f

            @test eltype(ps.layer_1.weight) == ftype
            @test eltype(ps.layer_1.bias) == ftype
            @test eltype(ps.layer_2.layer_1.weight) == ftype
            @test eltype(ps.layer_2.layer_1.bias) == ftype
            @test eltype(ps.layer_2.layer_2.weight) == ftype
            @test eltype(ps.layer_2.layer_2.bias) == ftype
            @test eltype(ps.layer_3.scale) == ftype
            @test eltype(ps.layer_3.bias) == ftype
            @test st.layer_1 == NamedTuple()
            @test st.layer_2.layer_1 == NamedTuple()
            @test st.layer_2.layer_2 == NamedTuple()
            @test eltype(st.layer_3.running_mean) == ftype
            @test eltype(st.layer_3.running_var) == ftype
            @test typeof(st.layer_3.training) == Val{true}
        end
    end
end
