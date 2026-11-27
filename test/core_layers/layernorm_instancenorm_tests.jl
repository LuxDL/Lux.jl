using Enzyme

include("../shared_testsetup.jl")

@testset "LayerNorm" begin
    rng = StableRNG(12345)

    ln_grad_configs = (
        ((3, 3, 3), true, identity),
        ((3, 3, 3), false, sigmoid),
        ((1, 3, 1), true, tanh),
        ((1, 3, 1), false, identity),
        ((3, 1, 3), true, sigmoid),
        ((3, 1, 3), false, tanh),
    )

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        x = aType(randn(rng, Float32, 3, 3, 3, 2))

        @testset for bshape in ((3, 3, 3), (1, 3, 1), (3, 1, 3)),
            affine in (true, false),
            act in (identity, sigmoid, tanh)

            ln = LayerNorm(bshape, act; affine)
            display(ln)
            ps, st = dev(Lux.setup(rng, ln))

            y, st_ = ln(x, ps, Lux.testmode(st))
            if act === identity
                @test mean(y) ≈ 0 atol = 1.0f-3
                @test std(y) ≈ 1 atol = 1.0f-2
            else
                @test size(y) == size(x)
            end

            if (bshape, affine, act) in ln_grad_configs
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

@testset "InstanceNorm" begin
    rng = StableRNG(12345)

    in_grad_configs = (
        (4, true, true, identity),
        (4, false, false, sigmoid),
        (3, true, false, tanh),
        (3, false, true, identity),
        (5, true, true, sigmoid),
        (5, false, false, tanh),
    )

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset "ndims(x) = $(ndims(x))" for x in (
            randn(rng, Float32, 3, 3, 3, 2),
            randn(rng, Float32, 3, 3, 2),
            randn(rng, Float32, 3, 3, 3, 3, 2),
        )
            x = aType(x)
            @testset for affine in (true, false),
                track_stats in (true, false),
                act in (identity, sigmoid, tanh)

                layer = InstanceNorm(3, act; affine, track_stats)
                display(layer)
                ps, st = dev(Lux.setup(rng, layer))

                y, st_ = layer(x, ps, Lux.testmode(st))
                @test size(y) == size(x)

                if (ndims(x), affine, track_stats, act) in in_grad_configs
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
