@testitem "BatchNorm" setup = [SharedTestSetup] tags = [:normalize_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        m = BatchNorm(2)
        x = [
            1.0f0 3.0f0 5.0f0
            2.0f0 4.0f0 6.0f0
        ] |> aType
        display(m)
        ps, st = Lux.setup(rng, m) |> dev

        @test Lux.parameterlength(m) == Lux.parameterlength(ps)
        @test Lux.statelength(m) == Lux.statelength(st)

        @test ps.bias == [0, 0] |> aType  # init_bias(2)
        @test ps.scale == [1, 1] |> aType  # init_scale(2)

        y, st_ = pullback(m, x, ps, st)[1]
        st_ = st_ |> CPUDevice()
        @test Array(y) ≈ [-1.22474 0 1.22474; -1.22474 0 1.22474] atol = 1.0e-5
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
        @test st_.running_mean ≈ reshape([0.3, 0.4], 2, 1)

        # julia> .1 .* var(x, dims = 2, corrected=false) .* (3 / 2).+ .9 .* [1., 1.]
        # 2×1 Array{Float64,2}:
        #  1.3
        #  1.3
        @test st_.running_var ≈
            0.1 .* var(Array(x); dims = 2, corrected = false) .* (3 / 2) .+ 0.9 .* [1.0, 1.0]

        st_ = Lux.testmode(st_) |> dev
        x_ = m(x, ps, st_)[1] |> CPUDevice()
        @test x_[1] ≈ (1 .- 0.3) / sqrt(1.3) atol = 1.0e-5

        skip_backends = VERSION ≥ v"1.11-" ? [AutoEnzyme()] : []
        skip_backends = vcat(skip_backends, [AutoFiniteDiff()])

        @jet m(x, ps, st)
        @test_gradients(
            sumabs2first, m, x, ps, st; atol = 1.0f-3, rtol = 1.0f-3, skip_backends
        )

        @testset for affine in (true, false)
            m = BatchNorm(2; affine, track_stats = false)
            x = [1.0f0 3.0f0 5.0f0; 2.0f0 4.0f0 6.0f0] |> aType
            display(m)
            ps, st = Lux.setup(rng, m) |> dev

            @jet m(x, ps, Lux.testmode(st))
            @test_gradients(sumabs2first, m, x, ps, st; atol = 1.0f-3, rtol = 1.0f-3)

            # with activation function
            m = BatchNorm(2, sigmoid; affine)
            x = [
                1.0f0 3.0f0 5.0f0
                2.0f0 4.0f0 6.0f0
            ] |> aType
            display(m)
            ps, st = Lux.setup(rng, m) |> dev

            y, st_ = m(x, ps, Lux.testmode(st))
            @test y ≈
                sigmoid.((x .- st_.running_mean) ./ sqrt.(st_.running_var .+ m.epsilon))
            @jet m(x, ps, Lux.testmode(st))
            @test_gradients(sumabs2first, m, x, ps, st; atol = 1.0f-3, rtol = 1.0f-3)

            m = BatchNorm(32; affine)
            x = randn(Float32, 416, 416, 32, 1) |> aType
            display(m)
            ps, st = Lux.setup(rng, m) |> dev
            m(x, ps, Lux.testmode(st))
            @jet m(x, ps, Lux.testmode(st))
        end
    end
end

@testitem "GroupNorm" setup = [SharedTestSetup] tags = [:normalize_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        squeeze(x) = dropdims(x; dims = tuple(findall(size(x) .== 1)...)) # To remove all singular dimensions

        m = GroupNorm(4, 2)
        sizes = (3, 4, 2)
        x = reshape(collect(1:prod(sizes)), sizes) |> aType

        display(m)
        x = Float32.(x)
        ps, st = Lux.setup(rng, m) |> dev
        @test Lux.parameterlength(m) == Lux.parameterlength(ps)
        @test Lux.statelength(m) == Lux.statelength(st)
        @test ps.bias == [0, 0, 0, 0] |> aType  # init_bias(32)
        @test ps.scale == [1, 1, 1, 1] |> aType # init_scale(32)

        @jet m(x, ps, st)
        @test_gradients(
            sumabs2first, m, x, ps, st; atol = 1.0f-3, rtol = 1.0f-3,
            enzyme_set_runtime_activity = true
        )

        @testset for affine in (true, false)
            m = GroupNorm(2, 2; affine)
            x = rand(rng, Float32, 3, 2, 1) |> aType
            display(m)
            ps, st = Lux.setup(rng, m) |> dev

            @jet m(x, ps, Lux.testmode(st))
            @test_gradients(
                sumabs2first, m, x, ps, st; atol = 1.0f-3, rtol = 1.0f-3,
                skip_backends = [AutoFiniteDiff()]
            )

            # with activation function
            m = GroupNorm(2, 2, sigmoid; affine)
            x = randn(rng, Float32, 3, 2, 1) |> aType
            display(m)
            ps, st = Lux.setup(rng, m) |> dev
            y, st_ = m(x, ps, Lux.testmode(st))
            @jet m(x, ps, Lux.testmode(st))
            @test_gradients(
                sumabs2first, m, x, ps, st; atol = 1.0f-3, rtol = 1.0f-3,
                skip_backends = [AutoFiniteDiff()]
            )

            m = GroupNorm(32, 16; affine)
            x = randn(rng, Float32, 416, 416, 32, 1) |> aType
            display(m)
            ps, st = Lux.setup(rng, m) |> dev
            m(x, ps, Lux.testmode(st))
            @jet m(x, ps, Lux.testmode(st))
        end

        @test_throws ArgumentError GroupNorm(5, 2)
    end
end

@testitem "WeightNorm" setup = [SharedTestSetup] tags = [:normalize_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset "Utils.norm_except" begin
            z = randn(rng, Float32, 3, 3, 4, 2) |> aType

            @test size(Lux.Utils.norm(z; dims = (1, 2))) == (1, 1, 4, 2)
            @test size(Lux.Utils.norm_except(z; dims = 1)) == (3, 1, 1, 1)
            @test Lux.Utils.norm_except(z; dims = 2) == Lux.Utils.norm(z; dims = (1, 3, 4))
            @test size(Lux.Utils.norm_except(z; dims = (1, 2))) == (3, 3, 1, 1)
            @test Lux.Utils.norm_except(z; dims = (1, 2)) == Lux.Utils.norm(z; dims = (3, 4))

            @jet Lux.Utils.norm_except(z)
            __f = z -> sum(Lux.Utils.norm_except(z; dims = (3, 4)))
            @jet __f(z)
        end

        @testset "Conv" begin
            c = Conv((3, 3), 3 => 3; init_bias = Lux.ones32)

            wn = WeightNorm(c, (:weight, :bias))
            display(wn)
            ps, st = Lux.setup(rng, wn) |> dev
            x = randn(rng, Float32, 3, 3, 3, 1) |> aType

            @jet wn(x, ps, st)
            @test_gradients(sumabs2first, wn, x, ps, st; atol = 1.0f-3, rtol = 1.0f-3)

            wn = WeightNorm(c, (:weight,))
            display(wn)
            ps, st = Lux.setup(rng, wn) |> dev
            x = randn(rng, Float32, 3, 3, 3, 1) |> aType

            @jet wn(x, ps, st)
            @test_gradients(sumabs2first, wn, x, ps, st; atol = 1.0f-3, rtol = 1.0f-3)

            wn = WeightNorm(c, (:weight, :bias), (2, 2))
            display(wn)
            ps, st = Lux.setup(rng, wn) |> dev
            x = randn(rng, Float32, 3, 3, 3, 1) |> aType

            @jet wn(x, ps, st)
            @test_gradients(sumabs2first, wn, x, ps, st; atol = 1.0f-3, rtol = 1.0f-3)

            wn = WeightNorm(c, (:weight,), (2,))
            display(wn)
            ps, st = Lux.setup(rng, wn) |> dev
            x = randn(rng, Float32, 3, 3, 3, 1) |> aType

            @jet wn(x, ps, st)
            @test_gradients(sumabs2first, wn, x, ps, st; atol = 1.0f-3, rtol = 1.0f-3)
        end

        @testset "Dense" begin
            d = Dense(3 => 3; init_bias = Lux.ones32)

            wn = WeightNorm(d, (:weight, :bias))
            display(wn)
            ps, st = Lux.setup(rng, wn) |> dev
            x = randn(rng, Float32, 3, 1) |> aType

            @jet wn(x, ps, st)
            @test_gradients(sumabs2first, wn, x, ps, st; atol = 1.0f-3, rtol = 1.0f-3)

            wn = WeightNorm(d, (:weight,))
            display(wn)
            ps, st = Lux.setup(rng, wn) |> dev
            x = randn(rng, Float32, 3, 1) |> aType

            @jet wn(x, ps, st)
            @test_gradients(sumabs2first, wn, x, ps, st; atol = 1.0f-3, rtol = 1.0f-3)

            wn = WeightNorm(d, (:weight, :bias), (2, 2))
            display(wn)
            ps, st = Lux.setup(rng, wn) |> dev
            x = randn(rng, Float32, 3, 1) |> aType

            @jet wn(x, ps, st)
            @test_gradients(sumabs2first, wn, x, ps, st; atol = 1.0f-3, rtol = 1.0f-3)

            wn = WeightNorm(d, (:weight,), (2,))
            display(wn)
            ps, st = Lux.setup(rng, wn) |> dev
            x = randn(rng, Float32, 3, 1) |> aType

            @jet wn(x, ps, st)
            @test_gradients(sumabs2first, wn, x, ps, st; atol = 1.0f-3, rtol = 1.0f-3)
        end

        # See https://github.com/LuxDL/Lux.jl/issues/95
        @testset "Normalizing Zero Parameters" begin
            c = Conv((3, 3), 3 => 3; init_bias = zeros32)

            wn = WeightNorm(c, (:weight, :bias))
            @test_throws ArgumentError Lux.setup(rng, wn)

            wn = WeightNorm(c, (:weight,))
            @test Lux.setup(rng, wn) isa Any

            c = Conv((3, 3), 3 => 3; init_bias = Lux.ones32)

            wn = WeightNorm(c, (:weight, :bias))
            @test Lux.setup(rng, wn) isa Any

            wn = WeightNorm(c, (:weight,))
            @test Lux.setup(rng, wn) isa Any
        end
    end
end

@testitem "LayerNorm" setup = [SharedTestSetup] tags = [:normalize_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        x = randn(rng, Float32, 3, 3, 3, 2) |> aType

        @testset for bshape in ((3, 3, 3), (1, 3, 1), (3, 1, 3))
            @testset for affine in (true, false)
                ln = LayerNorm(bshape; affine)
                display(ln)
                ps, st = Lux.setup(rng, ln) |> dev

                y, st_ = ln(x, ps, Lux.testmode(st))

                @test mean(y) ≈ 0 atol = 1.0f-3
                @test std(y) ≈ 1 atol = 1.0f-2

                @jet ln(x, ps, Lux.testmode(st))
                @test_gradients(
                    sumabs2first, ln, x, ps, st; atol = 1.0f-3,
                    rtol = 1.0f-3, skip_backends = [AutoFiniteDiff()]
                )

                @testset for act in (sigmoid, tanh)
                    ln = LayerNorm(bshape, act; affine)
                    display(ln)
                    ps, st = Lux.setup(rng, ln) |> dev

                    y, st_ = ln(x, ps, Lux.testmode(st))

                    @jet ln(x, ps, Lux.testmode(st))
                    @test_gradients(
                        sumabs2first, ln, x, ps, st; atol = 1.0f-3,
                        rtol = 1.0f-3, skip_backends = [AutoFiniteDiff()]
                    )
                end
            end
        end
    end
end

@testitem "InstanceNorm" setup = [SharedTestSetup] tags = [:normalize_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset "ndims(x) = $(ndims(x))" for x in (
                randn(rng, Float32, 3, 3, 3, 2), randn(rng, Float32, 3, 3, 2),
                randn(rng, Float32, 3, 3, 3, 3, 2),
            )
            x = x |> aType
            @testset for affine in (true, false), track_stats in (true, false)
                layer = InstanceNorm(3; affine, track_stats)
                display(layer)
                ps, st = Lux.setup(rng, layer) |> dev

                y, st_ = layer(x, ps, Lux.testmode(st))
                @jet layer(x, ps, Lux.testmode(st))
                @test_gradients(
                    sumabs2first, layer, x, ps, st; atol = 1.0f-3,
                    rtol = 1.0f-3, enzyme_set_runtime_activity = true,
                    skip_backends = [AutoFiniteDiff()]
                )

                @testset for act in (sigmoid, tanh)
                    layer = InstanceNorm(3, act; affine, track_stats)
                    display(layer)
                    ps, st = Lux.setup(rng, layer) |> dev

                    y, st_ = layer(x, ps, Lux.testmode(st))
                    @jet layer(x, ps, Lux.testmode(st))
                    @test_gradients(
                        sumabs2first, layer, x, ps, st; atol = 1.0f-3,
                        rtol = 1.0f-3, enzyme_set_runtime_activity = true,
                        skip_backends = [AutoFiniteDiff()]
                    )
                end
            end
        end
    end
end
