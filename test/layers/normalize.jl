using Lux, NNlib, Statistics, Zygote

include("../test_utils.jl")

rng = get_stable_rng(12345)

@testset "$mode: BatchNorm" for (mode, aType, device, ongpu) in MODES
    m = BatchNorm(2)
    x = [1.0f0 3.0f0 5.0f0
         2.0f0 4.0f0 6.0f0] |> aType
    display(m)
    ps, st = Lux.setup(rng, m) .|> device

    @test Lux.parameterlength(m) == Lux.parameterlength(ps)
    @test Lux.statelength(m) == Lux.statelength(st)

    @test ps.bias == [0, 0] |> aType  # init_bias(2)
    @test ps.scale == [1, 1] |> aType  # init_scale(2)

    y, st_ = pullback(m, x, ps, st)[1]
    st_ = st_ |> cpu
    @test check_approx(Array(y), [-1.22474 0 1.22474; -1.22474 0 1.22474]; atol=1.0e-5)
    # julia> x
    #  2×3 Array{Float64,2}:
    #  1.0  3.0  5.0
    #  2.0  4.0  6.0

    # mean of batch will be
    #  (1. + 3. + 5.) / 3 = 3
    #  (2. + 4. + 6.) / 3 = 4

    # ∴ update rule with momentum:
    #  .1 * 3 + 0 = .3
    #  .1 * 4 + 0 = .4
    @test check_approx(st_.running_mean, reshape([0.3, 0.4], 2, 1))

    # julia> .1 .* var(x, dims = 2, corrected=false) .* (3 / 2).+ .9 .* [1., 1.]
    # 2×1 Array{Float64,2}:
    #  1.3
    #  1.3
    @test check_approx(st_.running_var,
                       0.1 .* var(Array(x); dims=2, corrected=false) .* (3 / 2) .+
                       0.9 .* [1.0, 1.0])

    st_ = Lux.testmode(st_)
    x_ = m(x, ps, st_)[1] |> cpu
    @test check_approx(x_[1], (1 .- 0.3) / sqrt(1.3), atol=1.0e-5)

    @jet m(x, ps, st)
    __f = (x, ps) -> sum(first(m(x, ps, st)))
    @eval @test_gradients $__f $x $ps atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu skip_finite_differences=true

    for affine in (true, false)
        m = BatchNorm(2; affine, track_stats=false)
        x = [1.0f0 3.0f0 5.0f0; 2.0f0 4.0f0 6.0f0] |> aType
        display(m)
        ps, st = Lux.setup(rng, m) .|> device

        @jet m(x, ps, st)

        if affine
            __f = (x, ps) -> sum(first(m(x, ps, st)))
            @eval @test_gradients $__f $x $ps atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu skip_finite_differences=true
        else
            __f = x -> sum(first(m(x, ps, st)))
            @eval @test_gradients $__f $x atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu skip_finite_differences=true
        end

        # with activation function
        m = BatchNorm(2, sigmoid; affine)
        x = [1.0f0 3.0f0 5.0f0
             2.0f0 4.0f0 6.0f0] |> aType
        display(m)
        ps, st = Lux.setup(rng, m) .|> device
        st = Lux.testmode(st)
        y, st_ = m(x, ps, st)
        @test check_approx(y,
                           sigmoid.((x .- st_.running_mean) ./
                                    sqrt.(st_.running_var .+ m.epsilon)), atol=1.0e-7)

        @jet m(x, ps, st)

        if affine
            __f = (x, ps) -> sum(first(m(x, ps, st)))
            @eval @test_gradients $__f $x $ps atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu skip_finite_differences=true
        else
            __f = x -> sum(first(m(x, ps, st)))
            @eval @test_gradients $__f $x atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu skip_finite_differences=true
        end

        m = BatchNorm(32; affine)
        x = randn(Float32, 416, 416, 32, 1) |> aType
        display(m)
        ps, st = Lux.setup(rng, m)
        st = Lux.testmode(st)
        m(x, ps, st)
        @test (@allocated m(x, ps, st)) < 100_000_000

        @jet m(x, ps, st)
    end

    @testset "allow fast activation" begin
        layer = BatchNorm(10, tanh)
        @test layer.activation == tanh_fast
        layer = BatchNorm(10, tanh; allow_fast_activation=false)
        @test layer.activation == tanh
    end
end

@testset "$mode: GroupNorm" for (mode, aType, device, ongpu) in MODES
    # begin tests
    squeeze(x) = dropdims(x; dims=tuple(findall(size(x) .== 1)...)) # To remove all singular dimensions

    m = GroupNorm(4, 2; track_stats=true)
    sizes = (3, 4, 2)
    x = reshape(collect(1:prod(sizes)), sizes) |> aType

    display(m)
    x = Float32.(x)
    ps, st = Lux.setup(rng, m) .|> device
    @test Lux.parameterlength(m) == Lux.parameterlength(ps)
    @test Lux.statelength(m) == Lux.statelength(st)
    @test ps.bias == [0, 0, 0, 0] |> aType  # init_bias(32)
    @test ps.scale == [1, 1, 1, 1] |> aType # init_scale(32)

    y, st_ = pullback(m, x, ps, st)[1]
    y = y |> Array

    # julia> x
    # [:, :, 1]  =
    # 1.0  4.0  7.0  10.0
    # 2.0  5.0  8.0  11.0
    # 3.0  6.0  9.0  12.0
    #
    # [:, :, 2] =
    # 13.0  16.0  19.0  22.0
    # 14.0  17.0  20.0  23.0
    # 15.0  18.0  21.0  24.0
    #
    # mean will be
    # (1. + 2. + 3. + 4. + 5. + 6.) / 6 = 3.5
    # (7. + 8. + 9. + 10. + 11. + 12.) / 6 = 9.5
    #
    # (13. + 14. + 15. + 16. + 17. + 18.) / 6 = 15.5
    # (19. + 20. + 21. + 22. + 23. + 24.) / 6 = 21.5
    #
    # mean =
    # 3.5   15.5
    # 9.5   21.5
    #
    # ∴ update rule with momentum:
    # (1. - .1) * 0 + .1 * (3.5 + 15.5) / 2 = 0.95
    # (1. - .1) * 0 + .1 * (9.5 + 21.5) / 2 = 1.55
    @test check_approx(st_.running_mean, aType([0.95, 1.55]))
    n = prod(size(x)) ÷ m.groups ÷ size(x)[end]
    corr = n / (n - 1)
    z = reshape(x, 3, 2, 2, 2)
    variance = var(z; dims=(1, 2), corrected=false)
    @test check_approx(st_.running_var, 0.1 * corr * vec(mean(variance; dims=4)) .+ 0.9 * 1)

    st__ = Lux.testmode(st_)
    y, st__ = m(x, ps, st__)
    out = (z .- reshape(st_.running_mean, 1, 1, 2, 1)) ./
          sqrt.(reshape(st_.running_var, 1, 1, 2, 1) .+ 1.0f-5)
    @test check_approx(y, reshape(out, size(x)); atol=1.0e-5)

    @jet m(x, ps, st)
    __f = ps -> sum(first(m(x, ps, st)))
    @eval @test_gradients $__f $ps atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu

    for affine in (true, false)
        m = GroupNorm(2, 2; affine, track_stats=false)
        x = rand(rng, Float32, 3, 2, 1) |> aType
        display(m)
        ps, st = Lux.setup(rng, m) .|> device

        @jet m(x, ps, st)

        if affine
            __f = (x, ps) -> sum(first(m(x, ps, st)))
            @eval @test_gradients $__f $x $ps atol=1.0f-2 rtol=1.0f-2 gpu_testing=$ongpu skip_finite_differences=true
        else
            __f = x -> sum(first(m(x, ps, st)))
            @eval @test_gradients $__f $x atol=1.0f-2 rtol=1.0f-2 gpu_testing=$ongpu skip_finite_differences=true
        end

        # with activation function
        m = GroupNorm(2, 2, sigmoid; affine)
        x = randn(rng, Float32, 3, 2, 1) |> aType
        display(m)
        ps, st = Lux.setup(rng, m) .|> device
        st = Lux.testmode(st)
        y, st_ = m(x, ps, st)

        @jet m(x, ps, st)

        if affine
            __f = (x, ps) -> sum(first(m(x, ps, st)))
            @eval @test_gradients $__f $x $ps atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu skip_finite_differences=true
        else
            __f = x -> sum(first(m(x, ps, st)))
            @eval @test_gradients $__f $x atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu skip_finite_differences=true
        end

        m = GroupNorm(32, 16; affine)
        x = randn(rng, Float32, 416, 416, 32, 1) |> aType
        display(m)
        ps, st = Lux.setup(rng, m) .|> device
        st = Lux.testmode(st)
        m(x, ps, st)

        @test (@allocated m(x, ps, st)) < 100_000_000

        @jet m(x, ps, st)
    end

    @test_throws AssertionError GroupNorm(5, 2)

    @testset "allow fast activation" begin
        layer = GroupNorm(10, 2, tanh)
        @test layer.activation == tanh_fast
        layer = GroupNorm(10, 2, tanh; allow_fast_activation=false)
        @test layer.activation == tanh
    end

    # Deprecated Functionality (remove in v0.5)
    @test_deprecated GroupNorm(4, 2; track_stats=true)
    @test_deprecated GroupNorm(4, 2; track_stats=false, momentum=0.3f0)
end

@testset "$mode: WeightNorm" for (mode, aType, device, ongpu) in MODES
    @testset "_norm_except" begin
        z = randn(rng, Float32, 3, 3, 4, 2) |> aType

        @test size(Lux._norm(z; dims=(1, 2))) == (1, 1, 4, 2)
        @test size(Lux._norm_except(z; dims=1)) == (3, 1, 1, 1)
        @test Lux._norm_except(z; dims=2) == Lux._norm(z; dims=(1, 3, 4))
        @test size(Lux._norm_except(z; dims=(1, 2))) == (3, 3, 1, 1)
        @test Lux._norm_except(z; dims=(1, 2)) == Lux._norm(z; dims=(3, 4))

        @jet Lux._norm_except(z)
        __f = z -> sum(Lux._norm_except(z; dims=(3, 4)))
        @jet __f(z)
    end

    @testset "Conv" begin
        c = Conv((3, 3), 3 => 3; init_bias=Lux.ones32)

        wn = WeightNorm(c, (:weight, :bias))
        display(wn)
        ps, st = Lux.setup(rng, wn) .|> device
        x = randn(rng, Float32, 3, 3, 3, 1) |> aType

        @jet wn(x, ps, st)
        __f = ps -> sum(first(wn(x, ps, st)))
        @eval @test_gradients $__f $ps atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu skip_reverse_diff=true

        wn = WeightNorm(c, (:weight,))
        display(wn)
        ps, st = Lux.setup(rng, wn) .|> device
        x = randn(rng, Float32, 3, 3, 3, 1) |> aType

        @jet wn(x, ps, st)
        __f = (x, ps) -> sum(first(wn(x, ps, st)))
        @eval @test_gradients $__f $x $ps atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu skip_reverse_diff=true

        wn = WeightNorm(c, (:weight, :bias), (2, 2))
        display(wn)
        ps, st = Lux.setup(rng, wn) .|> device
        x = randn(rng, Float32, 3, 3, 3, 1) |> aType

        @jet wn(x, ps, st)
        __f = (x, ps) -> sum(first(wn(x, ps, st)))
        @eval @test_gradients $__f $x $ps atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu skip_reverse_diff=true

        wn = WeightNorm(c, (:weight,), (2,))
        display(wn)
        ps, st = Lux.setup(rng, wn) .|> device
        x = randn(rng, Float32, 3, 3, 3, 1) |> aType

        @jet wn(x, ps, st)
        __f = (x, ps) -> sum(first(wn(x, ps, st)))
        @eval @test_gradients $__f $x $ps atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu skip_reverse_diff=true
    end

    @testset "Dense" begin
        d = Dense(3 => 3; init_bias=Lux.ones32)

        wn = WeightNorm(d, (:weight, :bias))
        display(wn)
        ps, st = Lux.setup(rng, wn) .|> device
        x = randn(rng, Float32, 3, 1) |> aType

        @jet wn(x, ps, st)
        __f = ps -> sum(first(wn(x, ps, st)))
        @eval @test_gradients $__f $ps atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu

        wn = WeightNorm(d, (:weight,))
        display(wn)
        ps, st = Lux.setup(rng, wn) .|> device
        x = randn(rng, Float32, 3, 1) |> aType

        @jet wn(x, ps, st)
        __f = (x, ps) -> sum(first(wn(x, ps, st)))
        @eval @test_gradients $__f $x $ps atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu

        wn = WeightNorm(d, (:weight, :bias), (2, 2))
        display(wn)
        ps, st = Lux.setup(rng, wn) .|> device
        x = randn(rng, Float32, 3, 1) |> aType

        @jet wn(x, ps, st)
        __f = (x, ps) -> sum(first(wn(x, ps, st)))
        @eval @test_gradients $__f $x $ps atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu

        wn = WeightNorm(d, (:weight,), (2,))
        display(wn)
        ps, st = Lux.setup(rng, wn) .|> device
        x = randn(rng, Float32, 3, 1) |> aType

        @jet wn(x, ps, st)
        __f = (x, ps) -> sum(first(wn(x, ps, st)))
        @eval @test_gradients $__f $x $ps atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu
    end

    # See https://github.com/avik-pal/Lux.jl/issues/95
    @testset "Normalizing Zero Parameters" begin
        c = Conv((3, 3), 3 => 3)

        wn = WeightNorm(c, (:weight, :bias))
        @test_throws ArgumentError Lux.setup(rng, wn)

        wn = WeightNorm(c, (:weight,))
        @test_nowarn Lux.setup(rng, wn)

        c = Conv((3, 3), 3 => 3; init_bias=Lux.ones32)

        wn = WeightNorm(c, (:weight, :bias))
        @test_nowarn Lux.setup(rng, wn)

        wn = WeightNorm(c, (:weight,))
        @test_nowarn Lux.setup(rng, wn)
    end
end

@testset "$mode: LayerNorm" for (mode, aType, device, ongpu) in MODES
    x = randn(rng, Float32, 3, 3, 3, 2) |> aType

    for bshape in ((3, 3, 3), (1, 3, 1), (3, 1, 3))
        for affine in (true, false)
            ln = LayerNorm(bshape; affine)
            display(ln)
            ps, st = Lux.setup(rng, ln) .|> device

            y, st_ = ln(x, ps, st)

            @test check_approx(mean(y), 0; atol=1.0f-3, rtol=1.0f-3)
            @test check_approx(std(y), 1; atol=1.0f-2, rtol=1.0f-2)

            @jet ln(x, ps, st)

            if affine
                __f = (x, ps) -> sum(first(ln(x, ps, st)))
                @eval @test_gradients $__f $x $ps atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu skip_finite_differences=true
            else
                __f = x -> sum(first(ln(x, ps, st)))
                @eval @test_gradients $__f $x atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu skip_finite_differences=true
            end

            for act in (sigmoid, tanh)
                ln = LayerNorm(bshape, act; affine)
                display(ln)
                ps, st = Lux.setup(rng, ln) .|> device

                y, st_ = ln(x, ps, st)

                @jet ln(x, ps, st)

                if affine
                    __f = (x, ps) -> sum(first(ln(x, ps, st)))
                    @eval @test_gradients $__f $x $ps atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu skip_finite_differences=true
                else
                    __f = x -> sum(first(ln(x, ps, st)))
                    @eval @test_gradients $__f $x atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu skip_finite_differences=true
                end
            end
        end
    end

    @testset "allow fast activation" begin
        layer = LayerNorm((3, 1), tanh)
        @test layer.activation == tanh_fast
        layer = LayerNorm((3, 1), tanh; allow_fast_activation=false)
        @test layer.activation == tanh
    end
end

@testset "$mode: InstanceNorm" for (mode, aType, device, ongpu) in MODES
    for x in (randn(rng, Float32, 3, 3, 3, 2), randn(rng, Float32, 3, 3, 2),
              randn(rng, Float32, 3, 3, 3, 3, 2))
        x = x |> aType
        for affine in (true, false)
            layer = InstanceNorm(3; affine)
            display(layer)
            ps, st = Lux.setup(rng, layer) .|> device

            y, st_ = layer(x, ps, st)

            @jet layer(x, ps, st)

            if affine
                __f = (x, ps) -> sum(first(layer(x, ps, st)))
                @eval @test_gradients $__f $x $ps atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu skip_finite_differences=true
            else
                __f = x -> sum(first(layer(x, ps, st)))
                @eval @test_gradients $__f $x atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu skip_finite_differences=true
            end

            for act in (sigmoid, tanh)
                layer = InstanceNorm(3, act; affine)
                display(layer)
                ps, st = Lux.setup(rng, layer) .|> device

                y, st_ = layer(x, ps, st)

                @jet layer(x, ps, st)

                if affine
                    __f = (x, ps) -> sum(first(layer(x, ps, st)))
                    @eval @test_gradients $__f $x $ps atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu skip_finite_differences=true
                else
                    __f = x -> sum(first(layer(x, ps, st)))
                    @eval @test_gradients $__f $x atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu skip_finite_differences=true
                end
            end
        end
    end

    @testset "allow fast activation" begin
        layer = InstanceNorm(3, tanh)
        @test layer.activation == tanh_fast
        layer = InstanceNorm(3, tanh; allow_fast_activation=false)
        @test layer.activation == tanh
    end
end
