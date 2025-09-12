using Lux, StableRNGs, Test
using Flux: Flux

include("../../setup_modes.jl")

toluxpsst = FromFluxAdaptor(; preserve_ps_st=true)
tolux = FromFluxAdaptor()
toluxforce = FromFluxAdaptor(; force_preserve=true, preserve_ps_st=true)

@testset "$mode" for (mode, aType, dev, ongpu) in MODES
    @testset "Containers" begin
        @testset "Chain" begin
            for model in [
                dev(Flux.Chain(Flux.Dense(2 => 5), Flux.Dense(5 => 1))),
                dev(Flux.Chain(; l1=Flux.Dense(2 => 5), l2=Flux.Dense(5 => 1))),
            ]
                x = aType(rand(Float32, 2, 1))

                model_lux = toluxpsst(model)
                ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

                @test model(x) ≈ model_lux(x, ps, st)[1]

                model_lux = tolux(model)
                ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

                @test size(model_lux(x, ps, st)[1]) == (1, 1)
            end
        end

        @testset "Maxout" begin
            model = dev(Flux.Maxout(() -> Flux.Dense(2 => 5), 4))
            x = aType(rand(Float32, 2, 1))

            model_lux = toluxpsst(model)
            ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

            @test model(x) ≈ model_lux(x, ps, st)[1]

            model_lux = tolux(model)
            ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

            @test size(model_lux(x, ps, st)[1]) == (5, 1)
        end

        @testset "Skip Connection" begin
            model = dev(Flux.SkipConnection(Flux.Dense(2 => 2), +))
            x = aType(rand(Float32, 2, 1))

            model_lux = toluxpsst(model)
            ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

            @test model(x) ≈ model_lux(x, ps, st)[1]

            model_lux = tolux(model)
            ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

            @test size(model_lux(x, ps, st)[1]) == (2, 1)
        end

        @testset "Parallel" begin
            for model in [
                dev(Flux.Parallel(+, Flux.Dense(2 => 2), Flux.Dense(2 => 2))),
                dev(Flux.Parallel(+; l1=Flux.Dense(2 => 2), l2=Flux.Dense(2 => 2))),
            ]
                x = aType(rand(Float32, 2, 1))

                model_lux = toluxpsst(model)
                ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

                @test model(x) ≈ model_lux(x, ps, st)[1]

                model_lux = tolux(model)
                ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

                @test size(model_lux(x, ps, st)[1]) == (2, 1)
            end
        end

        @testset "Pairwise Fusion" begin
            model = dev(Flux.PairwiseFusion(+, Flux.Dense(2 => 2), Flux.Dense(2 => 2)))
            x = aType.((rand(Float32, 2, 1), rand(Float32, 2, 1)))

            model_lux = toluxpsst(model)
            ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

            @test all(model(x) .≈ model_lux(x, ps, st)[1])

            model_lux = tolux(model)
            ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

            @test all(size.(model_lux(x, ps, st)[1]) .== ((2, 1),))
        end
    end

    @testset "Linear" begin
        @testset "Dense" begin
            for model in [dev(Flux.Dense(2 => 4)), dev(Flux.Dense(2 => 4; bias=false))]
                x = aType(randn(Float32, 2, 4))

                model_lux = toluxpsst(model)
                ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

                @test model(x) ≈ model_lux(x, ps, st)[1]

                model_lux = tolux(model)
                ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

                @test size(model_lux(x, ps, st)[1]) == size(model(x))
            end
        end

        @testset "Scale" begin
            for model in [dev(Flux.Scale(2)), dev(Flux.Scale(2; bias=false))]
                x = aType(randn(Float32, 2, 4))

                model_lux = toluxpsst(model)
                ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

                @test model(x) ≈ model_lux(x, ps, st)[1]

                model_lux = tolux(model)
                ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

                @test size(model_lux(x, ps, st)[1]) == size(model(x))
            end
        end

        @testset "Bilinear" begin
            for model in [
                dev(Flux.Bilinear((2, 3) => 5)), dev(Flux.Bilinear((2, 3) => 5; bias=false))
            ]
                x = aType(randn(Float32, 2, 4))
                y = aType(randn(Float32, 3, 4))

                model_lux = toluxpsst(model)
                ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

                @test model(x, y) ≈ model_lux((x, y), ps, st)[1]

                model_lux = tolux(model)
                ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

                @test size(model_lux((x, y), ps, st)[1]) == size(model(x, y))
            end
        end

        @testset "Embedding" begin
            model = dev(Flux.Embedding(16 => 4))
            x = aType(rand(1:16, 2, 4))

            model_lux = toluxpsst(model)
            ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

            @test model(x) ≈ model_lux(x, ps, st)[1]

            model_lux = tolux(model)
            ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

            @test size(model_lux(x, ps, st)[1]) == (4, 2, 4)
        end
    end

    @testset "Convolutions" begin
        @testset "Conv" begin
            model = dev(Flux.Conv((3, 3), 1 => 2))
            x = aType(rand(Float32, 6, 6, 1, 4))

            model_lux = toluxpsst(model)
            ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

            @test model(x) ≈ model_lux(x, ps, st)[1]

            model = dev(Flux.Conv((3, 3), 1 => 2; pad=Flux.SamePad()))
            x = aType(rand(Float32, 6, 6, 1, 4))

            model_lux = toluxpsst(model)
            ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

            @test model(x) ≈ model_lux(x, ps, st)[1]

            model_lux = tolux(model)
            ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

            @test size(model_lux(x, ps, st)[1]) == size(model(x))
        end

        @testset "CrossCor" begin
            model = dev(Flux.CrossCor((3, 3), 1 => 2))
            x = aType(rand(Float32, 6, 6, 1, 4))

            model_lux = toluxpsst(model)
            ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

            @test model(x) ≈ model_lux(x, ps, st)[1]

            model = dev(Flux.CrossCor((3, 3), 1 => 2; pad=Flux.SamePad()))
            x = aType(rand(Float32, 6, 6, 1, 4))

            model_lux = toluxpsst(model)
            ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

            @test model(x) ≈ model_lux(x, ps, st)[1]

            model_lux = tolux(model)
            ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

            @test size(model_lux(x, ps, st)[1]) == size(model(x))
        end

        @testset "ConvTranspose" begin
            model = dev(Flux.ConvTranspose((3, 3), 1 => 2))
            x = aType(rand(Float32, 6, 6, 1, 4))

            model_lux = toluxpsst(model)
            ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

            @test model(x) ≈ model_lux(x, ps, st)[1]

            model = dev(Flux.ConvTranspose((3, 3), 1 => 2; pad=Flux.SamePad()))
            x = aType(rand(Float32, 6, 6, 1, 4))

            model_lux = toluxpsst(model)
            ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

            @test model(x) ≈ model_lux(x, ps, st)[1]

            model_lux = tolux(model)
            ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

            @test size(model_lux(x, ps, st)[1]) == size(model(x))
        end
    end

    @testset "Pooling" begin
        @testset "AdaptiveMaxPooling" begin
            model = dev(Flux.AdaptiveMaxPool((2, 2)))
            x = aType(rand(Float32, 6, 6, 1, 4))

            model_lux = toluxpsst(model)
            ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

            @test model(x) ≈ model_lux(x, ps, st)[1]
        end

        @testset "AdaptiveMeanPooling" begin
            model = dev(Flux.AdaptiveMeanPool((2, 2)))
            x = aType(rand(Float32, 6, 6, 1, 4))

            model_lux = toluxpsst(model)
            ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

            @test model(x) ≈ model_lux(x, ps, st)[1]
        end

        @testset "MaxPooling" begin
            model = dev(Flux.MaxPool((2, 2)))
            x = aType(rand(Float32, 6, 6, 1, 4))

            model_lux = toluxpsst(model)
            ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

            @test model(x) ≈ model_lux(x, ps, st)[1]
        end

        @testset "MeanPooling" begin
            model = dev(Flux.MeanPool((2, 2)))
            x = aType(rand(Float32, 6, 6, 1, 4))

            model_lux = toluxpsst(model)
            ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

            @test model(x) ≈ model_lux(x, ps, st)[1]
        end

        @testset "GlobalMaxPooling" begin
            model = dev(Flux.GlobalMaxPool())
            x = aType(rand(Float32, 6, 6, 1, 4))

            model_lux = toluxpsst(model)
            ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

            @test model(x) ≈ model_lux(x, ps, st)[1]
        end

        @testset "GlobalMeanPooling" begin
            model = dev(Flux.GlobalMeanPool())
            x = aType(rand(Float32, 6, 6, 1, 4))

            model_lux = toluxpsst(model)
            ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

            @test model(x) ≈ model_lux(x, ps, st)[1]
        end
    end

    @testset "Upsampling" begin
        @testset "Upsample" begin
            model = dev(Flux.Upsample(5))
            x = aType(rand(Float32, 2, 2, 2, 1))

            model_lux = tolux(model)
            ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

            @test size(model_lux(x, ps, st)[1]) == (10, 10, 2, 1)
            @test model(x) ≈ model_lux(x, ps, st)[1]
        end

        @testset "PixelShuffle" begin
            model = dev(Flux.PixelShuffle(2))
            x = aType(randn(Float32, 2, 2, 4, 1))

            model_lux = tolux(model)
            ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

            @test size(model_lux(x, ps, st)[1]) == (4, 4, 1, 1)
            @test model(x) ≈ model_lux(x, ps, st)[1]
        end
    end

    @testset "Recurrent" begin
        @testset "RNNCell" begin
            model = dev(Flux.RNNCell(2 => 3))
            @test_throws Lux.FluxModelConversionException tolux(model)
            @test_throws Lux.FluxModelConversionException toluxforce(model)
        end

        @testset "LSTMCell" begin
            model = dev(Flux.LSTMCell(2 => 3))
            @test_throws Lux.FluxModelConversionException tolux(model)
            @test_throws Lux.FluxModelConversionException toluxforce(model)
        end

        @testset "GRUCell" begin
            model = dev(Flux.GRUCell(2 => 3))
            @test_throws Lux.FluxModelConversionException tolux(model)
            @test_throws Lux.FluxModelConversionException toluxforce(model)
        end
    end

    @testset "Normalize" begin
        @testset "BatchNorm" begin
            model = dev(Flux.BatchNorm(2))
            x = aType(randn(Float32, 2, 4))

            model_lux = toluxpsst(model)
            ps, st = dev(Lux.setup(StableRNG(12345), model_lux))
            st = Lux.testmode(st)

            @test model(x) ≈ model_lux(x, ps, st)[1]

            x = aType(randn(Float32, 2, 2, 2, 1))

            @test model(x) ≈ model_lux(x, ps, st)[1]

            model_lux = toluxforce(model)
            ps, st = dev(Lux.setup(StableRNG(12345), model_lux))
            st = Lux.testmode(st)

            @test model(x) ≈ model_lux(x, ps, st)[1]

            model_lux = tolux(model)
            ps, st = dev(Lux.setup(StableRNG(12345), model_lux))
            st = Lux.testmode(st)

            @test size(model_lux(x, ps, st)[1]) == size(model(x))
        end

        @testset "GroupNorm" begin
            model = dev(Flux.GroupNorm(4, 2))
            x = aType(randn(Float32, 2, 2, 4, 1))

            model_lux = toluxpsst(model)
            ps, st = dev(Lux.setup(StableRNG(12345), model_lux))
            st = Lux.testmode(st)

            @test model(x) ≈ model_lux(x, ps, st)[1]

            model_lux = toluxforce(model)
            ps, st = dev(Lux.setup(StableRNG(12345), model_lux))
            st = Lux.testmode(st)

            @test model(x) ≈ model_lux(x, ps, st)[1]

            model_lux = tolux(model)
            ps, st = dev(Lux.setup(StableRNG(12345), model_lux))
            st = Lux.testmode(st)

            @test size(model_lux(x, ps, st)[1]) == size(model(x))
        end

        @testset "LayerNorm" begin
            model = dev(Flux.LayerNorm(4))
            x = aType(randn(Float32, 4, 4, 4, 1))

            model_lux = toluxpsst(model)
            ps, st = dev(Lux.setup(StableRNG(12345), model_lux))
            st = Lux.testmode(st)

            @test model(x) ≈ model_lux(x, ps, st)[1]
        end

        @testset "InstanceNorm" begin
            model = dev(Flux.InstanceNorm(4))
            x = aType(randn(Float32, 4, 4, 4, 1))

            model_lux = toluxpsst(model)
            ps, st = dev(Lux.setup(StableRNG(12345), model_lux))

            @test model(x) ≈ model_lux(x, ps, st)[1]
        end
    end

    @testset "Dropout" begin
        @testset "Dropout" begin
            model = tolux(Flux.Dropout(0.5f0))

            x = aType(randn(Float32, 2, 4))
            ps, st = dev(Lux.setup(StableRNG(12345), model))

            @test size(model(x, ps, st)[1]) == size(x)

            x = aType(randn(Float32, 2, 3, 4))
            ps, st = dev(Lux.setup(StableRNG(12345), model))

            @test size(model(x, ps, st)[1]) == size(x)
        end

        @testset "AlphaDropout" begin
            model = tolux(Flux.AlphaDropout(0.5))

            x = aType(randn(Float32, 2, 4))
            ps, st = dev(Lux.setup(StableRNG(12345), model))

            @test size(model(x, ps, st)[1]) == size(x)

            x = aType(randn(Float32, 2, 4, 3))
            ps, st = dev(Lux.setup(StableRNG(12345), model))

            @test size(model(x, ps, st)[1]) == size(x)
        end
    end

    @testset "Custom Layer" begin
        struct CustomFluxLayer
            weight
            bias
        end

        Flux.@layer CustomFluxLayer

        (c::CustomFluxLayer)(x) = c.weight .* x .+ c.bias

        c = dev(CustomFluxLayer(randn(10), randn(10)))
        x = aType(randn(10))

        c_lux = tolux(c)
        display(c_lux)
        ps, st = dev(Lux.setup(StableRNG(12345), c_lux))

        @test c(x) ≈ c_lux(x, ps, st)[1]
    end

    @testset "Functions" begin
        @test tolux(Flux.flatten) isa Lux.FlattenLayer
        @test tolux(identity) isa Lux.NoOpLayer
        @test tolux(+) isa Lux.WrappedFunction
    end
end
