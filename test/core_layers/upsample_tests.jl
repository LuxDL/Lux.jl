using NNlib

include("../shared_testsetup.jl")

@testset "Upsample" begin
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
            sizes = (nothing, (8, 8), (8, 4))
            scales = (nothing, 2, (2, 1))

            @testset for umode in modes, xsize in sizes, scale in scales
                xor(isnothing(xsize), isnothing(scale)) || continue

                layer = Upsample(umode; size=xsize, scale=scale)
                display(layer)
                ps, st = dev(Lux.setup(rng, layer))
                x = aType(rand(4, 4, 3, 2))

                @jet layer(x, ps, st)

                y, _ = layer(x, ps, st)
                if isnothing(scale)
                    @test size(y)[1:2] == xsize
                else
                    @test size(y)[1:2] == size(x)[1:2] .* scale
                end
                @test size(y)[3:4] == size(x)[3:4]

                if scale == 2
                    @test_gradients(
                        sumabs2first,
                        layer,
                        x,
                        ps,
                        st;
                        atol=1.0f-3,
                        rtol=1.0f-3,
                        ground_truth_eltype=Nothing
                    )
                end
            end

            sizes = (nothing, (8, 8, 8), (8, 4, 4))
            scales = (nothing, 2, (2, 1, 1), (2, 2, 1))

            @testset for umode in modes, xsize in sizes, scale in scales
                xor(isnothing(xsize), isnothing(scale)) || continue

                layer = Upsample(umode; size=xsize, scale=scale)
                display(layer)
                ps, st = dev(Lux.setup(rng, layer))
                x = aType(rand(4, 4, 4, 2, 3))

                @jet layer(x, ps, st)

                y, _ = layer(x, ps, st)

                if isnothing(scale)
                    @test size(y)[1:3] == xsize
                else
                    @test size(y)[1:3] == size(x)[1:3] .* scale
                end
                @test size(y)[4:5] == size(x)[4:5]

                if xsize == (8, 8, 8)
                    @test_gradients(
                        sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3
                    )
                end
            end
        end

        @testset "align_corners" begin
            # Bilinear
            x_bi = aType(randn(rng, Float32, 4, 4, 2, 2))
            for (layer_t, layer_f, nnlib_t, nnlib_f) in (
                (
                    Upsample(:bilinear; scale=2, align_corners=true),
                    Upsample(:bilinear; scale=2, align_corners=false),
                    NNlib.upsample_bilinear(x_bi, 2; align_corners=true),
                    NNlib.upsample_bilinear(x_bi, 2; align_corners=false),
                ),
                (
                    Upsample(:bilinear; size=(8, 8), align_corners=true),
                    Upsample(:bilinear; size=(8, 8), align_corners=false),
                    NNlib.upsample_bilinear(x_bi; size=(8, 8), align_corners=true),
                    NNlib.upsample_bilinear(x_bi; size=(8, 8), align_corners=false),
                ),
            )
                ps_t, st_t = dev(Lux.setup(rng, layer_t))
                ps_f, st_f = dev(Lux.setup(rng, layer_f))
                y_t, _ = layer_t(x_bi, ps_t, st_t)
                y_f, _ = layer_f(x_bi, ps_f, st_f)

                @test y_t == nnlib_t
                @test y_f == nnlib_f
                @test y_t != y_f

                @jet layer_t(x_bi, ps_t, st_t)
                @test_gradients(
                    sumabs2first,
                    layer_t,
                    x_bi,
                    ps_t,
                    st_t;
                    atol=1.0f-3,
                    rtol=1.0f-3,
                    ground_truth_eltype=Nothing,
                )
            end

            # Trilinear
            x_tri = aType(randn(rng, Float32, 4, 4, 4, 2, 2))
            for (layer_t, layer_f, nnlib_t, nnlib_f) in (
                (
                    Upsample(:trilinear; scale=2, align_corners=true),
                    Upsample(:trilinear; scale=2, align_corners=false),
                    NNlib.upsample_trilinear(x_tri, 2; align_corners=true),
                    NNlib.upsample_trilinear(x_tri, 2; align_corners=false),
                ),
                (
                    Upsample(:trilinear; size=(8, 8, 8), align_corners=true),
                    Upsample(:trilinear; size=(8, 8, 8), align_corners=false),
                    NNlib.upsample_trilinear(x_tri; size=(8, 8, 8), align_corners=true),
                    NNlib.upsample_trilinear(x_tri; size=(8, 8, 8), align_corners=false),
                ),
            )
                ps_t, st_t = dev(Lux.setup(rng, layer_t))
                ps_f, st_f = dev(Lux.setup(rng, layer_f))
                y_t, _ = layer_t(x_tri, ps_t, st_t)
                y_f, _ = layer_f(x_tri, ps_f, st_f)

                @test y_t == nnlib_t
                @test y_f == nnlib_f
                @test y_t != y_f

                @jet layer_t(x_tri, ps_t, st_t)
                @test_gradients(
                    sumabs2first,
                    layer_t,
                    x_tri,
                    ps_t,
                    st_t;
                    atol=1.0f-3,
                    rtol=1.0f-3,
                    ground_truth_eltype=Nothing,
                )
            end
        end
    end
end

@testset "PixelShuffle" begin
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
