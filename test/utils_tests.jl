@testitem "replicate" setup=[SharedTestSetup] tags=[:others] begin
    @testset "$mode" for (mode, aType, device, ongpu) in MODES
        _rng = get_default_rng(mode)
        @test randn(_rng, 10, 2) != randn(_rng, 10, 2)
        @test randn(Lux.replicate(_rng), 10, 2)==randn(Lux.replicate(_rng), 10, 2) broken=(mode ==
                                                                                           "amdgpu")
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

@testitem "ComponentArrays edge cases" tags=[:others] begin
    using ComponentArrays

    @test eltype(ComponentArray()) == Float32
    @test eltype(ComponentArray(NamedTuple())) == Float32
    @test eltype(ComponentArray{Float64}(NamedTuple())) == Float64

    @test eltype(ComponentArray(Any[:a, 1], (FlatAxis(),))) == Any
end

@testitem "Deprecations" tags=[:others] begin
    using Functors

    @test_deprecated Lux.disable_stacktrace_truncation!()
    @test_deprecated Lux.cpu(rand(2))
    @test_deprecated Lux.gpu(rand(2))

    model = NoOpLayer()
    @test_deprecated Lux.Experimental.StatefulLuxLayer(model, (;), (;))

    @test_deprecated Lux.Experimental.DebugLayer(model; location="model")
    dmodel = Lux.Experimental.DebugLayer(model; location="model")
    @test dmodel.location == KeyPath(:model)
end

@testitem "multigate" setup=[SharedTestSetup] tags=[:others] begin
    rng = StableRNG(12345)

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

    rng = StableRNG(12345)

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
        @test_nowarn display(ComponentArray(NamedTuple()))
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
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, device, ongpu) in MODES
        rnn = RNNCell(3 => 5; init_state=Lux.zeros32)
        x = randn(rng, Float32, 3, 2, 2)
        @test Lux._init_hidden_state(rng, rnn, view(device(x), :, 1, :)) ==
              aType(zeros(Float32, 5, 2))
    end
end

@testitem "FP Conversions" setup=[SharedTestSetup] tags=[:others] begin
    rng = StableRNG(12345)

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

            @test_throws ArgumentError (ps.|>f)
            @test_throws ArgumentError f.(ps)

            x = [1.0, 2.0, 3.0] |> aType
            @test eltype(f(x)) == ftype
            x = ComplexF64.([1, 2, 3]) |> aType
            @test eltype(f(x)) == complex(ftype)
        end
    end
end

@testitem "Edge Cases" tags=[:others] begin
    @test Lux.__size(nothing) === nothing
    @test Lux.__size(1) === ()
    @test Lux.__size(1.0) === ()
    @test Lux.__size([1, 2]) === (2,)

    struct ABC
        a
        b
    end

    abc = ABC(1, 2)

    @test_throws ErrorException Lux.__named_tuple(abc)
    @test_throws MethodError Lux._pairs(abc)

    Base.NamedTuple(abc::ABC) = (a=abc.a, b=abc.b)

    @test Lux.__named_tuple(abc) == (a=1, b=2)
    @test Lux._pairs(abc) == pairs((a=1, b=2))

    @test Lux._merge(1.0, []) == 1.0
    @test Lux._merge([], 1.0) == 1.0
    @test_throws ArgumentError Lux._merge([2.0], 1)
    @test Lux._merge(abc, abc) == (a=1, b=2)
end

@testitem "Recursive Utils" tags=[:others] begin
    using Functors

    struct functorABC{A, B}
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
    end

    @testset "recursive_make_zero" begin
        nt = (; a=1.0, b=rand(2), c=[rand(2), rand(4)], d=(rand(3), rand(5)),
            e=nothing, f=Val(2), g=functorABC(randn(3, 2), randn(2, 3)))

        nt_zero = Lux.recursive_make_zero(nt)

        @test nt_zero.e === nothing
        @test nt_zero.f === Val(2)
        for leaf in (nt_zero.a, nt_zero.b, nt_zero.c[1], nt_zero.c[2],
            nt_zero.d[1], nt_zero.d[2], nt_zero.g.a, nt_zero.g.b)
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

@testitem "Functors Compatibility" setup=[SharedTestSetup] tags=[:others] begin
    using Functors

    rng = StableRNG(12345)

    c = Parallel(+; chain=Chain(; dense_1=Dense(2 => 3), dense_2=Dense(3 => 5)),
        dense_3=Dense(5 => 1))

    @test_nowarn fmap(println, c)

    l = Dense(2 => 2)
    new_model = fmap(x -> l, c)
    @test new_model.layers.chain.layers.dense_1 == l
    @test new_model.layers.chain.layers.dense_2 == l
    @test new_model.layers.dense_3 == l
end

@testitem "Tracing AD: AoS to SoA" setup=[SharedTestSetup] tags=[:autodiff] begin
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
