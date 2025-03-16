@testitem "replicate" setup = [SharedTestSetup] tags = [:misc] begin
    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        _rng = get_default_rng(mode)
        @test randn(_rng, 10, 2) != randn(_rng, 10, 2)
        @test randn(Lux.replicate(_rng), 10, 2) == randn(Lux.replicate(_rng), 10, 2) broken = (
            mode == "amdgpu"
        )
    end
end

@testitem "istraining" tags = [:misc] begin
    using Static

    @test LuxOps.istraining(Val(true))
    @test !LuxOps.istraining(Val(false))
    @test !LuxOps.istraining((training=Val(false),))
    @test LuxOps.istraining((training=Val(true),))
    @test !LuxOps.istraining((no_training=1,))
    @test LuxOps.istraining((training=true,))
    @test !LuxOps.istraining((training=false,))
    @test LuxOps.istraining(static(true))
    @test !LuxOps.istraining(static(false))
end

@testitem "ComponentArrays edge cases" tags = [:misc] begin
    using ComponentArrays

    @test eltype(ComponentArray()) == Float32
    @test eltype(ComponentArray(NamedTuple())) == Float32
    @test eltype(ComponentArray{Float64}(NamedTuple())) == Float64

    @test eltype(ComponentArray(Any[:a, 1], (FlatAxis(),))) == Any
end

@testitem "multigate" setup = [SharedTestSetup] tags = [:misc] begin
    rng = StableRNG(12345)

    function bcast_multigate(x)
        x1, x2, x3 = LuxOps.multigate(x, Val(3))
        return sum(x1) + sum(x3 .+ x2 .^ 2)
    end

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        x = aType(randn(rng, 10, 1))
        x1, x2 = LuxOps.multigate(x, Val(2))

        @test x1 == x[1:5, :]
        @test x2 == x[6:10, :]

        @jet LuxOps.multigate(x, Val(2))

        x = aType(randn(rng, 10))
        x1, x2 = LuxOps.multigate(x, Val(2))

        @test x1 == x[1:5]
        @test x2 == x[6:10]

        @jet LuxOps.multigate(x, Val(2))

        x = aType(rand(6, 5))
        res, (dx,) = Zygote.withgradient(bcast_multigate, x)

        @jet LuxOps.multigate(x, Val(3))

        @test res ≈ sum(x[1:2, :]) + sum(x[5:6, :]) + sum(abs2, x[3:4, :])
        @test dx ≈ aType([ones(2, 5); Array(x[3:4, :] .* 2); ones(2, 5)])

        @test_gradients(bcast_multigate, x; atol=1.0f-3, rtol=1.0f-3)
    end
end

@testitem "ComponentArrays" setup = [SharedTestSetup] tags = [:misc] begin
    using Optimisers, Functors

    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        ps = (weight=randn(rng, 3, 4), bias=randn(rng, 4))
        p_flat, re = Optimisers.destructure(ps)
        ps_c = ComponentArray(ps)

        @test ps_c.weight == ps.weight
        @test ps_c.bias == ps.bias

        @test p_flat == getdata(ps_c)
        @test -p_flat == getdata(-ps_c)
        @test zero(p_flat) == getdata(zero(ps_c))

        @test similar(ps_c, 10) isa Any
        @test similar(ps_c) isa Any

        ps_c_f, ps_c_re = Functors.functor(ps_c)
        @test ps_c_f == ps
        @test ps_c_re(ps_c_f) == ps_c

        # Empty ComponentArray test
        @test display(ComponentArray(NamedTuple())) isa Any
        println()

        # Optimisers
        opt = Adam(0.001f0)
        ps_c = dev(ps_c)
        st_opt = Optimisers.setup(opt, ps_c)

        @test Optimisers.update(st_opt, ps_c, ps_c) isa Any
        @test Optimisers.update!(st_opt, ps_c, ps_c) isa Any
    end

    # Ref: https://github.com/LuxDL/Lux.jl/issues/243
    nn = Chain(Dense(4, 3), Dense(3, 2))
    ps, st = Lux.setup(rng, nn)

    l2reg(p) = sum(abs2, ComponentArray(p))
    @test length(Zygote.gradient(l2reg, ps)) == 1
end

@testitem "Utils.init_rnn_hidden_state" setup = [SharedTestSetup] tags = [:recurrent_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        rnn = RNNCell(3 => 5; init_state=Lux.zeros32)
        x = randn(rng, Float32, 3, 2, 2)
        @test Lux.Utils.init_rnn_hidden_state(rng, rnn, view(dev(x), :, 1, :)) ==
            aType(zeros(Float32, 5, 2))
    end
end

@testitem "FP Conversions" setup = [SharedTestSetup] tags = [:misc] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        model = Chain(
            Dense(1 => 16, relu), Chain(Dense(16 => 1), Dense(1 => 1)), BatchNorm(1)
        )

        for (f, ftype) in zip((f16, f32, f64), (Float16, Float32, Float64))
            ps, st = f(dev(Lux.setup(rng, model)))

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

            @test_throws ArgumentError (f.(ps))
            @test_throws ArgumentError f.(ps)

            x = aType([1.0, 2.0, 3.0])
            @test eltype(f(x)) == ftype
            x = aType(ComplexF64.([1, 2, 3]))
            @test eltype(f(x)) == complex(ftype)
        end
    end
end

@testitem "Edge Cases" tags = [:misc] begin
    @test Lux.Utils.size(nothing) === nothing
    @test Lux.Utils.size(1) == ()
    @test Lux.Utils.size(1.0) == ()
    @test Lux.Utils.size([1, 2]) == (2,)

    struct ABC
        a
        b
    end

    abc = ABC(1, 2)

    @test_throws ErrorException Lux.Utils.named_tuple(abc)
    @test_throws MethodError Lux.Utils.pairs(abc)

    Base.NamedTuple(abc::ABC) = (a=abc.a, b=abc.b)

    @test Lux.Utils.named_tuple(abc) == (a=1, b=2)
    @test Lux.Utils.pairs(abc) == pairs((a=1, b=2))

    @test Lux.Utils.merge(1.0, []) == 1.0
    @test Lux.Utils.merge([], 1.0) == 1.0
    @test_throws ArgumentError Lux.Utils.merge([2.0], 1)
    @test Lux.Utils.merge(abc, abc) == (a=1, b=2)
end

@testitem "Recursive Utils (Deprecated)" tags = [:misc] begin
    using Functors, Tracker, ReverseDiff, ForwardDiff

    struct functorABC{A,B}
        a::A
        b::B
    end

    @functor functorABC

    # These are not exclusive tests but they cases cases not covered by the other tests
    @testset "recursive_add!!" begin
        x = functorABC(randn(3, 2), randn(2, 3))
        x_c = deepcopy(x)
        y = functorABC(randn(3, 2), randn(2, 3))

        x = Lux.recursive_add!!(x, y)

        @test x.a ≈ x_c.a + y.a
        @test x.b ≈ x_c.b + y.b
        @test x.a != x_c.a
        @test x.b != x_c.b
    end

    @testset "recursive_eltype" begin
        @test Lux.recursive_eltype(randn(3, 2)) == Float64
        @test Lux.recursive_eltype(nothing) == Bool
        @test Lux.recursive_eltype(3.0) == Float64

        @testset "AD wrappers" begin
            x = randn(3, 2)
            @test Lux.recursive_eltype(x, Val(true)) == Float64

            x_wrapped = (
                ForwardDiff.Dual.(x),
                ForwardDiff.Dual(2.0),
                ReverseDiff.track.(x),
                ReverseDiff.track(2.0),
                ReverseDiff.track(x),
                Tracker.param.(x),
                Tracker.param(x),
                Tracker.param(2.0),
            )

            for x in x_wrapped
                @test Lux.recursive_eltype(x, Val(true)) == Float64
                @test Lux.recursive_eltype(x, Val(false)) == eltype(x)
            end
        end
    end

    @testset "recursive_make_zero" begin
        nt = (;
            a=1.0,
            b=rand(2),
            c=[rand(2), rand(4)],
            d=(rand(3), rand(5)),
            e=nothing,
            f=Val(2),
            g=functorABC(randn(3, 2), randn(2, 3)),
        )

        nt_zero = Lux.recursive_make_zero(nt)

        @test nt_zero.e === nothing
        @test nt_zero.f === Val(2)
        for leaf in (
            nt_zero.a,
            nt_zero.b,
            nt_zero.c[1],
            nt_zero.c[2],
            nt_zero.d[1],
            nt_zero.d[2],
            nt_zero.g.a,
            nt_zero.g.b,
        )
            @test iszero(leaf)
        end

        nt_ = Lux.recursive_make_zero!!(nt)

        @test nt_.e === nothing
        @test nt_.f === Val(2)
        for leaf in (nt_.a, nt_.b, nt_.c[1], nt_.c[2], nt_.d[1], nt_.d[2], nt_.g.a, nt_.g.b)
            @test iszero(leaf)
        end

        for leaf in (nt.b, nt.c[1], nt.c[2], nt.d[1], nt.d[2], nt.g.a, nt.g.b)
            @test iszero(leaf)
        end
        @test nt.a == 1.0
    end
end

@testitem "Functors Compatibility" setup = [SharedTestSetup] tags = [:misc] begin
    using Functors

    rng = StableRNG(12345)

    c = Parallel(
        +;
        chain=Chain(; dense_1=Dense(2 => 3), dense_2=Dense(3 => 5)),
        dense_3=Dense(5 => 1),
    )

    @test fmap(println, c) isa Any

    l = Dense(2 => 2)
    new_model = fmap(x -> l, c)
    @test new_model.layers.chain.layers.dense_1 == l
    @test new_model.layers.chain.layers.dense_2 == l
    @test new_model.layers.dense_3 == l
end

@testitem "Tracing AD: AoS to SoA" setup = [SharedTestSetup] tags = [:autodiff] begin
    using ReverseDiff, Tracker

    rng = StableRNG(1234)

    x = rand(rng, Float32, 1, 128)
    nn = Dense(1 => 1)
    ps, st = Lux.setup(rng, nn)

    x_t = Tracker.TrackedReal.(x)
    y_t = LuxCore.stateless_apply(nn, x_t, ps)
    @test y_t isa Tracker.TrackedArray

    y_t = first(nn(x_t, ps, st))
    @test y_t isa AbstractArray{<:Tracker.TrackedReal}

    x_t = ReverseDiff.TrackedReal.(x, zero(x))
    y_t = LuxCore.stateless_apply(nn, x_t, ps)
    @test y_t isa ReverseDiff.TrackedArray

    y_t = first(nn(x_t, ps, st))
    @test y_t isa AbstractArray{<:ReverseDiff.TrackedReal}
end
