@testitem "FromFluxAdaptor" setup=[SharedTestSetup] tags=[:others] begin
    import Flux

    from_flux = fdev(::Lux.LuxCPUDevice) = Flux.cpu
    fdev(::Lux.LuxCUDADevice) = Base.Fix1(Flux.gpu, Flux.FluxCUDAAdaptor())
    fdev(::Lux.LuxAMDGPUDevice) = Base.Fix1(Flux.gpu, Flux.FluxAMDAdaptor())

    toluxpsst = FromFluxAdaptor(; preserve_ps_st=true)
    tolux = FromFluxAdaptor()
    toluxforce = FromFluxAdaptor(; force_preserve=true, preserve_ps_st=true)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset "Containers" begin
            @testset "Chain" begin
                models = [Flux.Chain(Flux.Dense(2 => 5), Flux.Dense(5 => 1)),
                    Flux.Chain(; l1=Flux.Dense(2 => 5), l2=Flux.Dense(5 => 1))] .|>
                         fdev(dev)

                for model in models
                    x = rand(Float32, 2, 1) |> aType

                    model_lux = toluxpsst(model)
                    ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                    @test model(x) ≈ model_lux(x, ps, st)[1]

                    model_lux = tolux(model)
                    ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                    @test size(model_lux(x, ps, st)[1]) == (1, 1)
                end
            end

            @testset "Maxout" begin
                model = Flux.Maxout(() -> Flux.Dense(2 => 5), 4) |> fdev(dev)
                x = rand(Float32, 2, 1) |> aType

                model_lux = toluxpsst(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                @test model(x) ≈ model_lux(x, ps, st)[1]

                model_lux = tolux(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                @test size(model_lux(x, ps, st)[1]) == (5, 1)
            end

            @testset "Skip Connection" begin
                model = Flux.SkipConnection(Flux.Dense(2 => 2), +) |> fdev(dev)
                x = rand(Float32, 2, 1) |> aType

                model_lux = toluxpsst(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                @test model(x) ≈ model_lux(x, ps, st)[1]

                model_lux = tolux(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                @test size(model_lux(x, ps, st)[1]) == (2, 1)
            end

            @testset "Parallel" begin
                models = [Flux.Parallel(+, Flux.Dense(2 => 2), Flux.Dense(2 => 2)),
                    Flux.Parallel(+; l1=Flux.Dense(2 => 2), l2=Flux.Dense(2 => 2))] .|>
                         fdev(dev)

                for model in models
                    x = rand(Float32, 2, 1) |> aType

                    model_lux = toluxpsst(model)
                    ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                    @test model(x) ≈ model_lux(x, ps, st)[1]

                    model_lux = tolux(model)
                    ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                    @test size(model_lux(x, ps, st)[1]) == (2, 1)
                end
            end

            @testset "Pairwise Fusion" begin
                model = Flux.PairwiseFusion(+, Flux.Dense(2 => 2), Flux.Dense(2 => 2)) |>
                        fdev(dev)
                x = (rand(Float32, 2, 1), rand(Float32, 2, 1)) .|> aType

                model_lux = toluxpsst(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                @test all(model(x) .≈ model_lux(x, ps, st)[1])

                model_lux = tolux(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                @test all(size.(model_lux(x, ps, st)[1]) .== ((2, 1),))
            end
        end

        @testset "Linear" begin
            @testset "Dense" begin
                for model in [Flux.Dense(2 => 4) |> fdev(dev),
                    Flux.Dense(2 => 4; bias=false) |> fdev(dev)]
                    x = randn(Float32, 2, 4) |> aType

                    model_lux = toluxpsst(model)
                    ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                    @test model(x) ≈ model_lux(x, ps, st)[1]

                    model_lux = tolux(model)
                    ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                    @test size(model_lux(x, ps, st)[1]) == size(model(x))
                end
            end

            @testset "Scale" begin
                for model in [
                    Flux.Scale(2) |> fdev(dev), Flux.Scale(2; bias=false) |> fdev(dev)]
                    x = randn(Float32, 2, 4) |> aType

                    model_lux = toluxpsst(model)
                    ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                    @test model(x) ≈ model_lux(x, ps, st)[1]

                    model_lux = tolux(model)
                    ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                    @test size(model_lux(x, ps, st)[1]) == size(model(x))
                end
            end

            @testset "Bilinear" begin
                for model in [Flux.Bilinear((2, 3) => 5) |> fdev(dev),
                    Flux.Bilinear((2, 3) => 5; bias=false) |> fdev(dev)]
                    x = randn(Float32, 2, 4) |> aType
                    y = randn(Float32, 3, 4) |> aType

                    model_lux = toluxpsst(model)
                    ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                    @test model(x, y) ≈ model_lux((x, y), ps, st)[1]

                    model_lux = tolux(model)
                    ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                    @test size(model_lux((x, y), ps, st)[1]) == size(model(x, y))
                end
            end

            @testset "Embedding" begin
                model = Flux.Embedding(16 => 4) |> fdev(dev)
                x = rand(1:16, 2, 4) |> aType

                model_lux = toluxpsst(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                @test model(x) ≈ model_lux(x, ps, st)[1]

                model_lux = tolux(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                @test size(model_lux(x, ps, st)[1]) == (4, 2, 4)
            end
        end

        @testset "Convolutions" begin
            @testset "Conv" begin
                model = Flux.Conv((3, 3), 1 => 2) |> fdev(dev)
                x = rand(Float32, 6, 6, 1, 4) |> aType

                model_lux = toluxpsst(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                @test model(x) ≈ model_lux(x, ps, st)[1]

                model = Flux.Conv((3, 3), 1 => 2; pad=Flux.SamePad()) |> fdev(dev)
                x = rand(Float32, 6, 6, 1, 4) |> aType

                model_lux = toluxpsst(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                @test model(x) ≈ model_lux(x, ps, st)[1]

                model_lux = tolux(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                @test size(model_lux(x, ps, st)[1]) == size(model(x))
            end

            @testset "CrossCor" begin
                model = Flux.CrossCor((3, 3), 1 => 2) |> fdev(dev)
                x = rand(Float32, 6, 6, 1, 4) |> aType

                model_lux = toluxpsst(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                @test model(x) ≈ model_lux(x, ps, st)[1]

                model = Flux.CrossCor((3, 3), 1 => 2; pad=Flux.SamePad()) |> fdev(dev)
                x = rand(Float32, 6, 6, 1, 4) |> aType

                model_lux = toluxpsst(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                @test model(x) ≈ model_lux(x, ps, st)[1]

                model_lux = tolux(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                @test size(model_lux(x, ps, st)[1]) == size(model(x))
            end

            @testset "ConvTranspose" begin
                model = Flux.ConvTranspose((3, 3), 1 => 2) |> fdev(dev)
                x = rand(Float32, 6, 6, 1, 4) |> aType

                model_lux = toluxpsst(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                @test model(x) ≈ model_lux(x, ps, st)[1]

                model = Flux.ConvTranspose((3, 3), 1 => 2; pad=Flux.SamePad()) |> fdev(dev)
                x = rand(Float32, 6, 6, 1, 4) |> aType

                model_lux = toluxpsst(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                @test model(x) ≈ model_lux(x, ps, st)[1]

                model_lux = tolux(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                @test size(model_lux(x, ps, st)[1]) == size(model(x))
            end
        end

        @testset "Pooling" begin
            @testset "AdaptiveMaxPooling" begin
                model = Flux.AdaptiveMaxPool((2, 2)) |> fdev(dev)
                x = rand(Float32, 6, 6, 1, 4) |> aType

                model_lux = toluxpsst(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                @test model(x) ≈ model_lux(x, ps, st)[1]
            end

            @testset "AdaptiveMeanPooling" begin
                model = Flux.AdaptiveMeanPool((2, 2)) |> fdev(dev)
                x = rand(Float32, 6, 6, 1, 4) |> aType

                model_lux = toluxpsst(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                @test model(x) ≈ model_lux(x, ps, st)[1]
            end

            @testset "MaxPooling" begin
                model = Flux.MaxPool((2, 2)) |> fdev(dev)
                x = rand(Float32, 6, 6, 1, 4) |> aType

                model_lux = toluxpsst(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                @test model(x) ≈ model_lux(x, ps, st)[1]
            end

            @testset "MeanPooling" begin
                model = Flux.MeanPool((2, 2)) |> fdev(dev)
                x = rand(Float32, 6, 6, 1, 4) |> aType

                model_lux = toluxpsst(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                @test model(x) ≈ model_lux(x, ps, st)[1]
            end

            @testset "GlobalMaxPooling" begin
                model = Flux.GlobalMaxPool() |> fdev(dev)
                x = rand(Float32, 6, 6, 1, 4) |> aType

                model_lux = toluxpsst(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                @test model(x) ≈ model_lux(x, ps, st)[1]
            end

            @testset "GlobalMeanPooling" begin
                model = Flux.GlobalMeanPool() |> fdev(dev)
                x = rand(Float32, 6, 6, 1, 4) |> aType

                model_lux = toluxpsst(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                @test model(x) ≈ model_lux(x, ps, st)[1]
            end
        end

        @testset "Upsampling" begin
            @testset "Upsample" begin
                model = Flux.Upsample(5) |> fdev(dev)
                x = rand(Float32, 2, 2, 2, 1) |> aType

                model_lux = tolux(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                @test size(model_lux(x, ps, st)[1]) == (10, 10, 2, 1)
                @test model(x) ≈ model_lux(x, ps, st)[1]
            end

            @testset "PixelShuffle" begin
                model = Flux.PixelShuffle(2) |> fdev(dev)
                x = randn(Float32, 2, 2, 4, 1) |> aType

                model_lux = tolux(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                @test size(model_lux(x, ps, st)[1]) == (4, 4, 1, 1)
                @test model(x) ≈ model_lux(x, ps, st)[1]
            end
        end

        @testset "Recurrent" begin
            # @test_throws Lux.FluxModelConversionError transform(Flux.RNN(2 => 2))

            @testset "RNNCell" begin
                model = Flux.RNNCell(2 => 3) |> fdev(dev)
                x = rand(Float32, 2, 4) |> aType

                model_lux = tolux(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                @test size(model_lux(x, ps, st)[1][1]) == (3, 4)

                @test_throws Lux.FluxModelConversionError toluxforce(model)

                model_lux = toluxpsst(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                @test size(model_lux(x, ps, st)[1][1]) == (3, 4)
            end

            @testset "LSTMCell" begin
                model = Flux.LSTMCell(2 => 3) |> fdev(dev)
                x = rand(Float32, 2, 4) |> aType

                model_lux = tolux(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                @test size(model_lux(x, ps, st)[1][1]) == (3, 4)

                @test_throws Lux.FluxModelConversionError toluxforce(model)

                model_lux = toluxpsst(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                @test size(model_lux(x, ps, st)[1][1]) == (3, 4)
            end

            @testset "GRUCell" begin
                model = Flux.GRUCell(2 => 3) |> fdev(dev)
                x = rand(Float32, 2, 4) |> aType

                model_lux = tolux(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                @test size(model_lux(x, ps, st)[1][1]) == (3, 4)

                @test_throws Lux.FluxModelConversionError toluxforce(model)

                model_lux = toluxpsst(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                @test size(model_lux(x, ps, st)[1][1]) == (3, 4)
            end
        end

        @testset "Normalize" begin
            @testset "BatchNorm" begin
                model = Flux.BatchNorm(2) |> fdev(dev)
                x = randn(Float32, 2, 4) |> aType

                model_lux = toluxpsst(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev
                st = Lux.testmode(st)

                @test model(x) ≈ model_lux(x, ps, st)[1]

                x = randn(Float32, 2, 2, 2, 1) |> aType

                @test model(x) ≈ model_lux(x, ps, st)[1]

                model_lux = toluxforce(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev
                st = Lux.testmode(st)

                @test model(x) ≈ model_lux(x, ps, st)[1]

                model_lux = tolux(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev
                st = Lux.testmode(st)

                @test size(model_lux(x, ps, st)[1]) == size(model(x))
            end

            @testset "GroupNorm" begin
                model = Flux.GroupNorm(4, 2) |> fdev(dev)
                x = randn(Float32, 2, 2, 4, 1) |> aType

                model_lux = toluxpsst(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev
                st = Lux.testmode(st)

                @test model(x) ≈ model_lux(x, ps, st)[1]

                model_lux = toluxforce(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev
                st = Lux.testmode(st)

                @test model(x) ≈ model_lux(x, ps, st)[1]

                model_lux = tolux(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev
                st = Lux.testmode(st)

                @test size(model_lux(x, ps, st)[1]) == size(model(x))
            end

            @testset "LayerNorm" begin
                model = Flux.LayerNorm(4) |> fdev(dev)
                x = randn(Float32, 4, 4, 4, 1) |> aType

                model_lux = toluxpsst(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev
                st = Lux.testmode(st)

                @test model(x) ≈ model_lux(x, ps, st)[1]
            end

            @testset "InstanceNorm" begin
                model = Flux.InstanceNorm(4) |> fdev(dev)
                x = randn(Float32, 4, 4, 4, 1) |> aType

                model_lux = toluxpsst(model)
                ps, st = Lux.setup(StableRNG(12345), model_lux) .|> dev

                @test model(x) ≈ model_lux(x, ps, st)[1]
            end
        end

        @testset "Dropout" begin
            @testset "Dropout" begin
                model = tolux(Flux.Dropout(0.5f0))

                x = randn(Float32, 2, 4) |> aType
                ps, st = Lux.setup(StableRNG(12345), model) .|> dev

                @test size(model(x, ps, st)[1]) == size(x)

                x = randn(Float32, 2, 3, 4) |> aType
                ps, st = Lux.setup(StableRNG(12345), model) .|> dev

                @test size(model(x, ps, st)[1]) == size(x)
            end

            @testset "AlphaDropout" begin
                model = tolux(Flux.AlphaDropout(0.5))

                x = randn(Float32, 2, 4) |> aType
                ps, st = Lux.setup(StableRNG(12345), model) .|> dev

                @test size(model(x, ps, st)[1]) == size(x)

                x = randn(Float32, 2, 4, 3) |> aType
                ps, st = Lux.setup(StableRNG(12345), model) .|> dev

                @test size(model(x, ps, st)[1]) == size(x)
            end
        end

        @testset "Custom Layer" begin
            struct CustomFluxLayer
                weight
                bias
            end

            Flux.@functor CustomFluxLayer

            (c::CustomFluxLayer)(x) = c.weight .* x .+ c.bias

            c = CustomFluxLayer(randn(10), randn(10)) |> fdev(dev)
            x = randn(10) |> aType

            c_lux = tolux(c)
            display(c_lux)
            ps, st = Lux.setup(StableRNG(12345), c_lux) .|> dev

            @test c(x) ≈ c_lux(x, ps, st)[1]
        end

        @testset "Functions" begin
            @test tolux(Flux.flatten) isa Lux.FlattenLayer
            @test tolux(identity) isa Lux.NoOpLayer
            @test tolux(+) isa Lux.WrappedFunction{:direct_call}
        end

        @testset "Unsupported Layers" begin
            accum(h, x) = (h + x, x)
            rnn = Flux.Recur(accum, 0)

            @test_throws Lux.FluxModelConversionError tolux(rnn)
        end
    end
end
