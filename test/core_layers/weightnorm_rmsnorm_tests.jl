using Enzyme

include("../shared_testsetup.jl")

@testset "WeightNorm" begin
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

@testset "RMSNorm" begin
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
