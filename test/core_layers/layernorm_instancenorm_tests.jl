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

        @testset "Dimension Expansion: Transformer shape & 1D input" begin
            # Transformer-shaped input: (features, seq_len, batch)
            x_trans = aType(randn(rng, Float32, 3, 4, 2))
            for affine in (true, false)
                ln_trans = LayerNorm((3,); dims=1, affine)
                display(ln_trans)
                ps_t, st_t = dev(Lux.setup(rng, ln_trans))

                y_t, st_t_ = ln_trans(x_trans, ps_t, Lux.testmode(st_t))
                @test size(y_t) == (3, 4, 2)
                @test mean(y_t; dims=1) ≈ zeros(Float32, 1, 4, 2) atol = 1.0f-3
                @test std(y_t; dims=1, corrected=false) ≈ ones(Float32, 1, 4, 2) atol =
                    1.0f-2

                @jet ln_trans(x_trans, ps_t, Lux.testmode(st_t))
                @test_gradients(
                    sumabs2first,
                    ln_trans,
                    x_trans,
                    ps_t,
                    st_t;
                    atol=1.0f-3,
                    rtol=1.0f-3,
                    skip_backends=[AutoFiniteDiff()]
                )
            end

            # 1D vector input
            x_1d = aType(randn(rng, Float32, 3))
            for affine in (true, false)
                ln_1d = LayerNorm((3,); affine)
                display(ln_1d)
                ps_1d, st_1d = dev(Lux.setup(rng, ln_1d))

                y_1d, st_1d_ = ln_1d(x_1d, ps_1d, Lux.testmode(st_1d))
                @test size(y_1d) == (3,)
                @test mean(y_1d) ≈ 0.0f0 atol = 1.0f-3
                @test std(y_1d; corrected=false) ≈ 1.0f0 atol = 1.0f-2

                @jet ln_1d(x_1d, ps_1d, Lux.testmode(st_1d))
                @test_gradients(
                    sumabs2first,
                    ln_1d,
                    x_1d,
                    ps_1d,
                    st_1d;
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
