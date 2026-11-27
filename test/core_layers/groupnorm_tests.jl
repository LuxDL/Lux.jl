using Enzyme

include("../shared_testsetup.jl")

@testset "GroupNorm" begin
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
            x = aType(randn(rng, Float32, 32, 32, 32, 1))
            display(m)
            ps, st = dev(Lux.setup(rng, m))
            m(x, ps, Lux.testmode(st))
            @jet m(x, ps, Lux.testmode(st))
        end

        @test_throws AssertionError GroupNorm(5, 2)
    end
end
