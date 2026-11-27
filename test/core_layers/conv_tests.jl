include("../shared_testsetup.jl")

@testset "Conv" begin
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

                if (
                    (k == (1,) && kwarg == (; stride=1)) ||
                    (k == (2, 3) && kwarg == (; dilation=(1, 1), stride=1)) ||
                    (k == (1, 2, 3) && kwarg == (; stride=3))
                )
                    @jet layer(x, ps, st)
                    @test_gradients(
                        sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3
                    )
                end
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

@testset "Conv(cross_correlation=true)" begin
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

                if (
                    (k == (1,) && kwarg == (; stride=1)) ||
                    (k == (2, 3) && kwarg == (; dilation=(1, 1), stride=1)) ||
                    (k == (1, 2, 3) && kwarg == (; stride=3)) ||
                    (k == (2,) && kwarg == (; stride=1, use_bias=false))
                )
                    @jet layer(x, ps, st)
                    @test_gradients(
                        sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3
                    )
                end
            end
        end
    end
end
