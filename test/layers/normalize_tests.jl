@testitem "BatchNorm" setup = [SharedTestSetup] tags = [:normalize_layers] begin
    using Enzyme

    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        m = BatchNorm(2)
        x = aType([
            1.0f0 3.0f0 5.0f0
            2.0f0 4.0f0 6.0f0
        ])
        display(m)
        ps, st = dev(Lux.setup(rng, m))

        @test Lux.parameterlength(m) == Lux.parameterlength(ps)
        @test Lux.statelength(m) == Lux.statelength(st)

        @test ps.bias == aType([0, 0])  # init_bias(2)
        @test ps.scale == aType([1, 1])  # init_scale(2)

        y, st_ = pullback(m, x, ps, st)[1]
        st_ = CPUDevice()(st_)
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
            0.1 .* var(Array(x); dims=2, corrected=false) .* (3 / 2) .+ 0.9 .* [1.0, 1.0]

        st_ = dev(Lux.testmode(st_))
        x_ = CPUDevice()(m(x, ps, st_)[1])
        @test x_[1] ≈ (1 .- 0.3) / sqrt(1.3) atol = 1.0e-5

        # chunking will cause incorrect gradients
        skip_backends = [AutoFiniteDiff(), AutoForwardDiff()]

        @jet m(x, ps, st)
        @test_gradients(sumabs2first, m, x, ps, st; atol=1.0f-3, rtol=1.0f-3, skip_backends)

        @testset for affine in (true, false)
            m = BatchNorm(2; affine, track_stats=false)
            x = aType([1.0f0 3.0f0 5.0f0; 2.0f0 4.0f0 6.0f0])
            display(m)
            ps, st = dev(Lux.setup(rng, m))

            @jet m(x, ps, Lux.testmode(st))
            @test_gradients(
                sumabs2first, m, x, ps, st; atol=1.0f-3, rtol=1.0f-3, skip_backends
            )

            # with activation function
            m = BatchNorm(2, sigmoid; affine)
            x = aType([
                1.0f0 3.0f0 5.0f0
                2.0f0 4.0f0 6.0f0
            ])
            display(m)
            ps, st = dev(Lux.setup(rng, m))

            y, st_ = m(x, ps, Lux.testmode(st))
            @test y ≈
                sigmoid.((x .- st_.running_mean) ./ sqrt.(st_.running_var .+ m.epsilon))
            @jet m(x, ps, Lux.testmode(st))
            @test_gradients(
                sumabs2first, m, x, ps, st; atol=1.0f-3, rtol=1.0f-3, skip_backends
            )

            m = BatchNorm(32; affine)
            x = aType(randn(Float32, 416, 416, 32, 1))
            display(m)
            ps, st = dev(Lux.setup(rng, m))
            m(x, ps, Lux.testmode(st))
            @jet m(x, ps, Lux.testmode(st))
        end
    end
end

@testitem "GroupNorm" setup = [SharedTestSetup] tags = [:normalize_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        squeeze(x) = dropdims(x; dims=tuple(findall(size(x) .== 1)...)) # To remove all singular dimensions

        m = GroupNorm(4, 2)
        sizes = (3, 4, 2)
        x = aType(reshape(collect(1:prod(sizes)), sizes))

        display(m)
        x = Float32.(x)
        ps, st = dev(Lux.setup(rng, m))
        @test Lux.parameterlength(m) == Lux.parameterlength(ps)
        @test Lux.statelength(m) == Lux.statelength(st)
        @test ps.bias == aType([0, 0, 0, 0])  # init_bias(32)
        @test ps.scale == aType([1, 1, 1, 1]) # init_scale(32)

        @jet m(x, ps, st)
        @test_gradients(sumabs2first, m, x, ps, st; atol=1.0f-3, rtol=1.0f-3,)

        @testset for affine in (true, false)
            m = GroupNorm(2, 2; affine)
            x = aType(rand(rng, Float32, 3, 2, 1))
            display(m)
            ps, st = dev(Lux.setup(rng, m))

            @jet m(x, ps, Lux.testmode(st))
            @test_gradients(
                sumabs2first,
                m,
                x,
                ps,
                st;
                atol=1.0f-3,
                rtol=1.0f-3,
                skip_backends=[AutoFiniteDiff()]
            )

            # with activation function
            m = GroupNorm(2, 2, sigmoid; affine)
            x = aType(randn(rng, Float32, 3, 2, 1))
            display(m)
            ps, st = dev(Lux.setup(rng, m))
            y, st_ = m(x, ps, Lux.testmode(st))
            @jet m(x, ps, Lux.testmode(st))
            @test_gradients(
                sumabs2first,
                m,
                x,
                ps,
                st;
                atol=1.0f-3,
                rtol=1.0f-3,
                skip_backends=[AutoFiniteDiff()]
            )

            m = GroupNorm(32, 16; affine)
            x = aType(randn(rng, Float32, 416, 416, 32, 1))
            display(m)
            ps, st = dev(Lux.setup(rng, m))
            m(x, ps, Lux.testmode(st))
            @jet m(x, ps, Lux.testmode(st))
        end

        @test_throws AssertionError GroupNorm(5, 2)
    end
end

@testitem "WeightNorm" setup = [SharedTestSetup] tags = [:normalize_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset "Utils.norm_except" begin
            z = aType(randn(rng, Float32, 3, 3, 4, 2))

            @test size(Lux.Utils.norm(z; dims=(1, 2))) == (1, 1, 4, 2)
            @test size(Lux.Utils.norm_except(z; dims=1)) == (3, 1, 1, 1)
            @test Lux.Utils.norm_except(z; dims=2) == Lux.Utils.norm(z; dims=(1, 3, 4))
            @test size(Lux.Utils.norm_except(z; dims=(1, 2))) == (3, 3, 1, 1)
            @test Lux.Utils.norm_except(z; dims=(1, 2)) == Lux.Utils.norm(z; dims=(3, 4))

            @jet Lux.Utils.norm_except(z)
            __f = z -> sum(Lux.Utils.norm_except(z; dims=(3, 4)))
            @jet __f(z)
        end

        @testset "Conv" begin
            c = Conv((3, 3), 3 => 3; init_bias=Lux.ones32)

            wn = WeightNorm(c, (:weight, :bias))
            display(wn)
            ps, st = dev(Lux.setup(rng, wn))
            x = aType(randn(rng, Float32, 3, 3, 3, 1))

            @jet wn(x, ps, st)
            @test_gradients(sumabs2first, wn, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

            wn = WeightNorm(c, (:weight,))
            display(wn)
            ps, st = dev(Lux.setup(rng, wn))
            x = aType(randn(rng, Float32, 3, 3, 3, 1))

            @jet wn(x, ps, st)
            @test_gradients(sumabs2first, wn, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

            wn = WeightNorm(c, (:weight, :bias), (2, 2))
            display(wn)
            ps, st = dev(Lux.setup(rng, wn))
            x = aType(randn(rng, Float32, 3, 3, 3, 1))

            @jet wn(x, ps, st)
            @test_gradients(sumabs2first, wn, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

            wn = WeightNorm(c, (:weight,), (2,))
            display(wn)
            ps, st = dev(Lux.setup(rng, wn))
            x = aType(randn(rng, Float32, 3, 3, 3, 1))

            @jet wn(x, ps, st)
            @test_gradients(sumabs2first, wn, x, ps, st; atol=1.0f-3, rtol=1.0f-3)
        end

        @testset "Dense" begin
            d = Dense(3 => 3; init_bias=Lux.ones32)

            wn = WeightNorm(d, (:weight, :bias))
            display(wn)
            ps, st = dev(Lux.setup(rng, wn))
            x = aType(randn(rng, Float32, 3, 1))

            @jet wn(x, ps, st)
            @test_gradients(sumabs2first, wn, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

            wn = WeightNorm(d, (:weight,))
            display(wn)
            ps, st = dev(Lux.setup(rng, wn))
            x = aType(randn(rng, Float32, 3, 1))

            @jet wn(x, ps, st)
            @test_gradients(sumabs2first, wn, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

            wn = WeightNorm(d, (:weight, :bias), (2, 2))
            display(wn)
            ps, st = dev(Lux.setup(rng, wn))
            x = aType(randn(rng, Float32, 3, 1))

            @jet wn(x, ps, st)
            @test_gradients(sumabs2first, wn, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

            wn = WeightNorm(d, (:weight,), (2,))
            display(wn)
            ps, st = dev(Lux.setup(rng, wn))
            x = aType(randn(rng, Float32, 3, 1))

            @jet wn(x, ps, st)
            @test_gradients(sumabs2first, wn, x, ps, st; atol=1.0f-3, rtol=1.0f-3)
        end

        # See https://github.com/LuxDL/Lux.jl/issues/95
        @testset "Normalizing Zero Parameters" begin
            c = Conv((3, 3), 3 => 3; init_bias=zeros32)

            wn = WeightNorm(c, (:weight, :bias))
            @test_throws ArgumentError Lux.setup(rng, wn)

            wn = WeightNorm(c, (:weight,))
            @test Lux.setup(rng, wn) isa Any

            c = Conv((3, 3), 3 => 3; init_bias=Lux.ones32)

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
        x = aType(randn(rng, Float32, 3, 3, 3, 2))

        @testset for bshape in ((3, 3, 3), (1, 3, 1), (3, 1, 3))
            @testset for affine in (true, false)
                ln = LayerNorm(bshape; affine)
                display(ln)
                ps, st = dev(Lux.setup(rng, ln))

                y, st_ = ln(x, ps, Lux.testmode(st))

                @test mean(y) ≈ 0 atol = 1.0f-3
                @test std(y) ≈ 1 atol = 1.0f-2

                @jet ln(x, ps, Lux.testmode(st))
                @test_gradients(
                    sumabs2first,
                    ln,
                    x,
                    ps,
                    st;
                    atol=1.0f-3,
                    rtol=1.0f-3,
                    skip_backends=[AutoFiniteDiff()]
                )

                @testset for act in (sigmoid, tanh)
                    ln = LayerNorm(bshape, act; affine)
                    display(ln)
                    ps, st = dev(Lux.setup(rng, ln))

                    y, st_ = ln(x, ps, Lux.testmode(st))

                    @jet ln(x, ps, Lux.testmode(st))
                    @test_gradients(
                        sumabs2first,
                        ln,
                        x,
                        ps,
                        st;
                        atol=1.0f-3,
                        rtol=1.0f-3,
                        skip_backends=[AutoFiniteDiff()]
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
            randn(rng, Float32, 3, 3, 3, 2),
            randn(rng, Float32, 3, 3, 2),
            randn(rng, Float32, 3, 3, 3, 3, 2),
        )
            x = aType(x)
            @testset for affine in (true, false), track_stats in (true, false)
                layer = InstanceNorm(3; affine, track_stats)
                display(layer)
                ps, st = dev(Lux.setup(rng, layer))

                y, st_ = layer(x, ps, Lux.testmode(st))
                @jet layer(x, ps, Lux.testmode(st))
                @test_gradients(
                    sumabs2first,
                    layer,
                    x,
                    ps,
                    st;
                    atol=1.0f-3,
                    rtol=1.0f-3,
                    skip_backends=[AutoFiniteDiff()]
                )

                @testset for act in (sigmoid, tanh)
                    layer = InstanceNorm(3, act; affine, track_stats)
                    display(layer)
                    ps, st = dev(Lux.setup(rng, layer))

                    y, st_ = layer(x, ps, Lux.testmode(st))
                    @jet layer(x, ps, Lux.testmode(st))
                    @test_gradients(
                        sumabs2first,
                        layer,
                        x,
                        ps,
                        st;
                        atol=1.0f-3,
                        rtol=1.0f-3,
                        skip_backends=[AutoFiniteDiff()]
                    )
                end
            end
        end
    end
end

@testitem "RMSNorm" setup = [SharedTestSetup] tags = [:normalize_layers] begin
    rng = StableRNG(12345)

    @testset "Basic Construction" begin
        l = RMSNorm((4,))
        @test l.normalized_shape == (4,)
        @test l.epsilon ≈ 1.0f-5
        @test Lux.has_affine(l)

        @testset "Parameter Initialization" begin
            ps = Lux.initialparameters(rng, l)
            @test haskey(ps, :scale)
            @test size(ps.scale) == (4,)
            @test all(ps.scale .== 1.0f0)
        end

        @testset "Parameter Initialization (non-affine)" begin
            l_noaff = RMSNorm((4,); affine=false)
            ps_noaff = Lux.initialparameters(rng, l_noaff)
            @test isempty(ps_noaff)
        end
    end

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset "affine" begin
            l = RMSNorm((4,); affine=true)
            ps, st = dev(Lux.setup(rng, l))

            copyto!(ps.scale, randn(rng, Float32, 4))

            x = aType(randn(rng, Float32, 4, 5))
            y, st = l(x, ps, st)
            @test size(y) == size(x)
            @test st == NamedTuple()

            # RMS normalization: check mean and variance
            inv_rms = inv.(sqrt.(mean(abs2, x; dims=1)))
            norm_x = x .* inv_rms
            norm_x_affine = norm_x .* ps.scale
            @test y ≈ norm_x_affine
        end

        @testset "non-affine" begin
            l = RMSNorm((4,); affine=false)
            ps, st = dev(Lux.setup(rng, l))

            x = aType(randn(rng, Float32, 4, 5))
            y, st = l(x, ps, st)
            @test size(y) == size(x)
            @test st == NamedTuple()

            # RMS normalization: check mean and variance
            inv_rms = inv.(sqrt.(mean(abs2, x; dims=1)))
            norm_x = x .* inv_rms
            @test y ≈ norm_x
        end

        @testset "different normalized_shape" begin
            l = RMSNorm((2, 2))
            ps, st = dev(Lux.setup(rng, l))

            copyto!(ps.scale, randn(rng, Float32, 2, 2))

            x = aType(randn(rng, Float32, 2, 2, 3))
            y, st = l(x, ps, st)
            @test size(y) == size(x)
            @test st == NamedTuple()

            # RMS normalization: check mean and variance
            inv_rms = inv.(sqrt.(mean(abs2, x; dims=1:2)))
            norm_x = (x .* inv_rms) .* ps.scale
            @test y ≈ norm_x
        end
    end
end
