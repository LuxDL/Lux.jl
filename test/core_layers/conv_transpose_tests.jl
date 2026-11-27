include("../shared_testsetup.jl")

@testset "ConvTranspose" begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset for cross_correlation in (true, false)
            x = aType(randn(Float32, 5, 5, 1, 1))
            layer = Conv((3, 3), 1 => 1)
            ps, st = dev(Lux.setup(rng, layer))
            y = layer(x, ps, st)[1]

            layer = ConvTranspose((3, 3), 1 => 1; cross_correlation)
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))

            @jet layer(y, ps, st)

            x_hat1 = layer(y, ps, st)[1]

            layer = ConvTranspose((3, 3), 1 => 1; use_bias=false, cross_correlation)
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))

            @jet layer(y, ps, st)

            x_hat2 = layer(y, ps, st)[1]

            @test size(x_hat1) == size(x_hat2) == size(x)

            layer = ConvTranspose((3, 3), 1 => 1; cross_correlation)
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))
            x = aType(rand(Float32, 5, 5, 1, 1))

            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

            x = aType(rand(Float32, 5, 5, 2, 4))
            layer = ConvTranspose((3, 3), 2 => 3; cross_correlation)
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))

            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

            # test ConvTranspose supports groups argument
            x = aType(randn(Float32, 10, 10, 2, 3))
            layer1 = ConvTranspose((3, 3), 2 => 4; pad=SamePad(), cross_correlation)
            display(layer1)
            ps1, st1 = dev(Lux.setup(rng, layer1))
            @test size(ps1.weight) == (3, 3, 4, 2)
            @test size(layer1(x, ps1, st1)[1]) == (10, 10, 4, 3)

            layer2 = ConvTranspose(
                (3, 3), 2 => 4; groups=2, pad=SamePad(), cross_correlation
            )
            display(layer2)
            ps2, st2 = dev(Lux.setup(rng, layer2))

            @test size(ps2.weight) == (3, 3, 2, 2)
            @test size(layer1(x, ps1, st1)[1]) == size(layer2(x, ps2, st2)[1])
            @test_gradients(sumabs2first, layer1, x, ps1, st1; atol=1.0f-3, rtol=1.0f-3)
            @test_gradients(sumabs2first, layer2, x, ps2, st2; atol=1.0f-3, rtol=1.0f-3)

            x = aType(randn(Float32, 10, 2, 1))
            layer = ConvTranspose((3,), 2 => 4; pad=SamePad(), groups=2, cross_correlation)
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))

            @jet layer(x, ps, st)

            @test size(layer(x, ps, st)[1]) == (10, 4, 1)
            @test length(ps.weight) == 3 * (2 * 4) / 2

            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

            x = aType(randn(Float32, 10, 11, 4, 2))
            layer = ConvTranspose(
                (3, 5), 4 => 4; pad=SamePad(), groups=4, cross_correlation
            )
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))

            @jet layer(x, ps, st)

            @test size(layer(x, ps, st)[1]) == (10, 11, 4, 2)
            @test length(ps.weight) == (3 * 5) * (4 * 4) / 4
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

            x = aType(randn(Float32, 10, 11, 4, 2))
            layer = ConvTranspose(
                (3, 5), 4 => 4, tanh; pad=SamePad(), groups=4, cross_correlation
            )
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))

            @jet layer(x, ps, st)
            @test size(layer(x, ps, st)[1]) == (10, 11, 4, 2)
            @test length(ps.weight) == (3 * 5) * (4 * 4) / 4
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

            x = aType(randn(Float32, 10, 11, 12, 3, 2))
            layer = ConvTranspose(
                (3, 5, 3), 3 => 6; pad=SamePad(), groups=3, cross_correlation
            )
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))

            @jet layer(x, ps, st)
            @test size(layer(x, ps, st)[1]) == (10, 11, 12, 6, 2)
            @test length(ps.weight) == (3 * 5 * 3) * (3 * 6) / 3

            x = aType(randn(Float32, 10, 11, 12, 3, 2))
            layer = ConvTranspose(
                (3, 5, 3), 3 => 6, tanh; pad=SamePad(), groups=3, cross_correlation
            )
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))

            @jet layer(x, ps, st)
            @test size(layer(x, ps, st)[1]) == (10, 11, 12, 6, 2)
            @test length(ps.weight) == (3 * 5 * 3) * (3 * 6) / 3

            @test occursin(
                "groups=2",
                sprint(show, ConvTranspose((3, 3), 2 => 4; groups=2, cross_correlation)),
            )
            @test occursin(
                "2 => 4",
                sprint(show, ConvTranspose((3, 3), 2 => 4; groups=2, cross_correlation)),
            )

            @testset "SamePad size mismatch LuxDL/Lux.jl#534" begin
                layer = ConvTranspose(
                    (3,), 2 => 1; pad=SamePad(), stride=2, cross_correlation
                )
                display(layer)
                x = aType(ones(Float32, 2, 2, 1))
                ps, st = dev(Lux.setup(rng, layer))

                y = first(layer(x, ps, st))
                @test size(y) == (4, 1, 1)
                @jet layer(x, ps, st)
            end

            @testset "Catch Channel Mismatch Early: LuxDL/Lux.jl#455" begin
                layer = ConvTranspose((4, 4), 42 => 16; stride=2, pad=1, cross_correlation)

                x = aType(randn(Float32, 28, 28, 42, 3))
                ps, st = dev(Lux.setup(rng, layer))

                @test layer(x, ps, st) isa Any

                x = aType(randn(Float32, 28, 28, 46, 3))

                @test_throws DimensionMismatch layer(x, ps, st)

                x = aType(randn(Float32, 28, 28, 23, 3))

                @test_throws DimensionMismatch layer(x, ps, st)
            end

            @testset "with Output Padding" begin
                m1 = ConvTranspose((3, 5), 3 => 6; stride=3, cross_correlation)
                display(m1)
                m2 = ConvTranspose(
                    (3, 5), 3 => 6; stride=3, outpad=(1, 0), cross_correlation
                )
                display(m2)

                ps1, st1 = dev(Lux.setup(rng, m1))
                ps2, st2 = dev(Lux.setup(rng, m2))

                x = aType(randn(Float32, 10, 11, 3, 2))
                @test size(m1(x, ps1, st1)[1])[1:2] .+ (1, 0) ==
                    size(m2(x, ps2, st2)[1])[1:2]

                m1 = ConvTranspose((3, 5, 3), 3 => 6; stride=3, cross_correlation)
                display(m1)
                m2 = ConvTranspose(
                    (3, 5, 3), 3 => 6; stride=3, outpad=(1, 0, 1), cross_correlation
                )
                display(m2)

                ps1, st1 = dev(Lux.setup(rng, m1))
                ps2, st2 = dev(Lux.setup(rng, m2))

                x = aType(randn(Float32, 10, 11, 12, 3, 2))

                @test size(m1(x, ps1, st1)[1])[1:3] .+ (1, 0, 1) ==
                    size(m2(x, ps2, st2)[1])[1:3]
            end
        end
    end
end
