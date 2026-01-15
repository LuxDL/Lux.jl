@testitem "Conv" setup = [SharedTestSetup] tags = [:core_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset "Grouped Conv" begin
            x = aType(rand(rng, Float32, 4, 6, 1))
            layer = Conv((3,), 6 => 2; groups=2)
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))

            @test size(ps.weight) == (3, 3, 2)
            @test size(layer(x, ps, st)[1]) == (2, 2, 1)
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

            x = aType(rand(rng, Float32, 4, 4, 6, 1))
            layer = Conv((3, 3), 6 => 2; groups=2)
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))

            @test size(ps.weight) == (3, 3, 3, 2)
            @test size(layer(x, ps, st)[1]) == (2, 2, 2, 1)
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

            x = aType(rand(rng, Float32, 4, 4, 4, 6, 1))
            layer = Conv((3, 3, 3), 6 => 2; groups=2)
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))

            @test size(ps.weight) == (3, 3, 3, 3, 2)
            @test size(layer(x, ps, st)[1]) == (2, 2, 2, 2, 1)
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

            # Test that we cannot ask for non-integer multiplication factors
            @test_throws DimensionMismatch Conv((2, 2), 3 => 10; groups=2)
            @test_throws DimensionMismatch Conv((2, 2), 2 => 9; groups=2)

            @testset "Segfault Test LuxDL/Lux.jl#386" begin
                layer = Conv((5,), 32 => 32, tanh; groups=32)
                display(layer)
                x = aType(rand(rng, Float32, 16, 32, 1))
                ps, st = dev(Lux.setup(rng, layer))

                @jet layer(x, ps, st)
                @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)
            end
        end

        @testset "Asymmetric Padding" begin
            layer = Conv((3, 3), 1 => 1, relu; pad=(0, 1, 1, 2))
            display(layer)
            x = aType(ones(Float32, 28, 28, 1, 1))
            ps, st = dev(Lux.setup(rng, layer))

            ps.weight .= 1.0
            ps.bias .= 0.0

            y_hat = Array(layer(x, ps, st)[1][:, :, 1, 1])
            @test size(y_hat) == (27, 29)
            @test check_approx(y_hat[1, 1], 6.0)
            @test check_approx(y_hat[2, 2], 9.0)
            @test check_approx(y_hat[end, 1], 4.0)
            @test check_approx(y_hat[1, end], 3.0)
            @test check_approx(y_hat[1, end - 1], 6.0)
            @test check_approx(y_hat[end, end], 2.0)

            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)
        end

        @testset "Variable BitWidth Parameters FluxML/Flux.jl#1421" begin
            layer = Conv(
                (5, 5),
                10 => 20,
                identity;
                init_weight=(rng, dims...) -> aType(randn(rng, Float64, dims...)),
                init_bias=(rng, dims...) -> aType(randn(rng, Float16, dims...)),
            )
            display(layer)
            ps, st = Lux.setup(rng, layer)
            @test ps.weight isa aType{Float64,4}
            @test ps.bias isa aType{Float16,1}
        end

        @testset "Depthwise Conv" begin
            x = aType(randn(rng, Float32, 4, 4, 3, 2))

            layer = Conv((2, 2), 3 => 15; groups=3)
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))

            @test Lux.parameterlength(layer) == Lux.parameterlength(ps)
            @test size(layer(x, ps, st)[1], 3) == 15
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

            layer = Conv((2, 2), 3 => 9; groups=3)
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))

            @test size(layer(x, ps, st)[1], 3) == 9
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

            layer = Conv((2, 2), 3 => 9; groups=3, use_bias=false)
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))

            @test Lux.parameterlength(layer) == Lux.parameterlength(ps)
            @test size(layer(x, ps, st)[1], 3) == 9
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

            # Test that we cannot ask for non-integer multiplication factors
            @test_throws DimensionMismatch Conv((2, 2), 3 => 10; groups=3)
        end

        @testset "Conv SamePad kernelsize $k" for k in ((1,), (2,), (3,), (2, 3), (1, 2, 3))
            x = aType(ones(Float32, (k .+ 3)..., 1, 1))

            @testset "Kwargs: $kwarg" for kwarg in (
                (; stride=1), (; dilation=max.(k .÷ 2, 1), stride=1), (; stride=3)
            )
                layer = Conv(k, 1 => 1; pad=Lux.SamePad(), kwarg...)
                display(layer)
                ps, st = dev(Lux.setup(rng, layer))

                layer(x, ps, st)
                if kwarg.stride == 1
                    @test size(layer(x, ps, st)[1]) == size(x)
                else
                    @test size(layer(x, ps, st)[1])[1:(end - 2)] ==
                        cld.(size(x)[1:(end - 2)], kwarg.stride)
                end

                @jet layer(x, ps, st)
                @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)
            end
        end

        @testset "Conv with non quadratic window FluxML/Flux.jl#700" begin
            x = zeros(Float32, 7, 7, 1, 1)
            x[4, 4, 1, 1] = 1
            x = aType(x)

            layer = Conv((3, 3), 1 => 1; use_bias=false)
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))

            y = aType(zeros(eltype(ps.weight), 5, 5, 1, 1))
            y[2:(end - 1), 2:(end - 1), 1, 1] = ps.weight

            @test y ≈ layer(x, ps, st)[1] rtol = 1.0e-3 atol = 1.0e-3
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

            layer = Conv((3, 1), 1 => 1; use_bias=false)
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))

            y = aType(zeros(eltype(ps.weight), 5, 7, 1, 1))
            y[2:(end - 1), 4, 1, 1] = ps.weight

            @test y ≈ layer(x, ps, st)[1] rtol = 1.0e-3 atol = 1.0e-3
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

            layer = Conv((1, 3), 1 => 1; use_bias=false)
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))

            y = aType(zeros(eltype(ps.weight), 7, 5, 1, 1))
            y[4, 2:(end - 1), 1, 1] = ps.weight
            @test y ≈ layer(x, ps, st)[1] rtol = 1.0e-3 atol = 1.0e-3
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

            layer = Conv((1, 3), 1 => 1; init_weight=Lux.glorot_normal, use_bias=false)
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))

            y = aType(zeros(eltype(ps.weight), 7, 5, 1, 1))
            y[4, 2:(end - 1), 1, 1] = ps.weight

            @test y ≈ layer(x, ps, st)[1] rtol = 1.0e-3 atol = 1.0e-3
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)
        end
    end
end

@testitem "Upsample" setup = [SharedTestSetup] tags = [:core_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset "Construction" begin
            @test Upsample(:nearest; scale=2) isa Any
            @test Upsample(:nearest; size=(64, 64)) isa Any
            @test Upsample(:bilinear; scale=2) isa Any
            @test Upsample(:bilinear; size=(64, 64)) isa Any
            @test Upsample(:trilinear; scale=2) isa Any
            @test Upsample(:trilinear; size=(64, 64)) isa Any

            @test_throws AssertionError Upsample(:linear; scale=2)
            @test_throws ArgumentError Upsample(:nearest; scale=2, size=(64, 64))
            @test_throws ArgumentError Upsample(:nearest)

            @test Upsample(2) isa Any
            @test Upsample(2, :nearest) isa Any
        end

        @testset "Size Correctness" begin
            # NNlib is checking algorithmic correctness. So we should just verify correct
            # function call
            modes = (:nearest, :bilinear, :trilinear)
            sizes = (nothing, (64, 64), (64, 32))
            scales = (nothing, 2, (2, 1))

            @testset for umode in modes, xsize in sizes, scale in scales
                xor(isnothing(xsize), isnothing(scale)) || continue

                layer = Upsample(umode; size=xsize, scale=scale)
                display(layer)
                ps, st = dev(Lux.setup(rng, layer))
                x = aType(rand(32, 32, 3, 4))

                @jet layer(x, ps, st)

                y, _ = layer(x, ps, st)
                if isnothing(scale)
                    @test size(y)[1:2] == xsize
                else
                    @test size(y)[1:2] == size(x)[1:2] .* scale
                end
                @test size(y)[3:4] == size(x)[3:4]

                @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)
            end

            sizes = (nothing, (64, 64, 64), (64, 32, 128))
            scales = (nothing, 2, (2, 1, 1), (2, 2, 1))

            @testset for umode in modes, xsize in sizes, scale in scales
                xor(isnothing(xsize), isnothing(scale)) || continue

                layer = Upsample(umode; size=xsize, scale=scale)
                display(layer)
                ps, st = dev(Lux.setup(rng, layer))
                x = aType(rand(32, 32, 32, 3, 4))

                @jet layer(x, ps, st)

                y, _ = layer(x, ps, st)

                if isnothing(scale)
                    @test size(y)[1:3] == xsize
                else
                    @test size(y)[1:3] == size(x)[1:3] .* scale
                end
                @test size(y)[4:5] == size(x)[4:5]

                @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3,)
            end
        end
    end
end

@testitem "PixelShuffle" setup = [SharedTestSetup] tags = [:core_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        layer = PixelShuffle(2)
        display(layer)
        ps, st = dev(Lux.setup(rng, layer))
        x = aType(rand(rng, Float32, 3, 6, 3))

        y, st_ = layer(x, ps, st)
        @test y isa aType{Float32,3}
        @test size(y) == (6, 3, 3)
        @jet layer(x, ps, st)
        @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0e-3, rtol=1.0e-3)

        layer = PixelShuffle(3)
        display(layer)
        ps, st = dev(Lux.setup(rng, layer))
        x = aType(rand(Float32, 3, 4, 9, 3))

        y, st_ = layer(x, ps, st)
        @test y isa aType{Float32,4}
        @test size(y) == (9, 12, 1, 3)
        @jet layer(x, ps, st)
        @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0e-3, rtol=1.0e-3)
    end
end

@testitem "Conv(cross_correlation=true)" setup = [SharedTestSetup] tags = [:core_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset "Asymmetric Padding" begin
            layer = Conv((3, 3), 1 => 1, relu; pad=(0, 1, 1, 2), cross_correlation=true)
            display(layer)
            x = aType(ones(Float32, 28, 28, 1, 1))
            ps, st = dev(Lux.setup(rng, layer))

            ps.weight .= 1.0
            ps.bias .= 0.0

            y_hat = Array(layer(x, ps, st)[1][:, :, 1, 1])
            @test size(y_hat) == (27, 29)
            @test check_approx(y_hat[1, 1], 6.0)
            @test check_approx(y_hat[2, 2], 9.0)
            @test check_approx(y_hat[end, 1], 4.0)
            @test check_approx(y_hat[1, end], 3.0)
            @test check_approx(y_hat[1, end - 1], 6.0)
            @test check_approx(y_hat[end, end], 2.0)
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0e-3, rtol=1.0e-3)
        end

        @testset "Variable BitWidth Parameters FluxML/Flux.jl#1421" begin
            layer = Conv(
                (5, 5),
                10 => 20,
                identity;
                init_weight=(rng, dims...) -> aType(randn(rng, Float64, dims...)),
                init_bias=(rng, dims...) -> aType(randn(rng, Float16, dims...)),
                cross_correlation=true,
            )
            display(layer)
            ps, st = Lux.setup(rng, layer)
            @test ps.weight isa aType{Float64,4}
            @test ps.bias isa aType{Float16,1}
        end

        @testset "SamePad kernelsize $k" for k in ((1,), (2,), (3,), (2, 3), (1, 2, 3))
            x = aType(ones(Float32, (k .+ 3)..., 1, 1))

            @testset "Kwargs: $kwarg" for kwarg in (
                (; stride=1),
                (; dilation=max.(k .÷ 2, 1), stride=1),
                (; stride=3),
                (; stride=1, use_bias=false),
            )
                layer = Conv(k, 1 => 1; pad=Lux.SamePad(), kwarg..., cross_correlation=true)
                display(layer)
                ps, st = dev(Lux.setup(rng, layer))

                layer(x, ps, st)
                if kwarg.stride == 1
                    @test size(layer(x, ps, st)[1]) == size(x)
                else
                    @test size(layer(x, ps, st)[1])[1:(end - 2)] ==
                        cld.(size(x)[1:(end - 2)], kwarg.stride)
                end

                @jet layer(x, ps, st)
                @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)
            end
        end
    end
end

@testitem "ConvTranspose" setup = [SharedTestSetup] tags = [:core_layers] begin
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
