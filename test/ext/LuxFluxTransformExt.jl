import Flux
using Lux, Test

fdevice(::typeof(LuxCPUDevice)) = Flux.cpu
fdevice(::typeof(LuxCUDADevice)) = Flux.gpu

include("../test_utils.jl")

@testset "$mode: LuxFluxTransformExt" for (mode, aType, device, ongpu) in MODES
    @testset "Containers" begin
        @testset "Chain" begin
            model = Flux.Chain(Flux.Dense(2 => 5), Flux.Dense(5 => 1)) |> fdevice(device)
            x = rand(Float32, 2, 1) |> aType

            model_lux = transform(model; preserve_ps_st=true)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

            @test model(x) ≈ model_lux(x, ps, st)[1]

            model_lux = transform(model)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

            @test size(model_lux(x, ps, st)[1]) == (1, 1)
        end

        @testset "Maxout" begin
            model = Flux.Maxout(() -> Flux.Dense(2 => 5), 4) |> fdevice(device)
            x = rand(Float32, 2, 1) |> aType

            model_lux = transform(model; preserve_ps_st=true)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

            @test model(x) ≈ model_lux(x, ps, st)[1]

            model_lux = transform(model)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

            @test size(model_lux(x, ps, st)[1]) == (5, 1)
        end

        @testset "Skip Connection" begin
            model = Flux.SkipConnection(Flux.Dense(2 => 2), +) |> fdevice(device)
            x = rand(Float32, 2, 1) |> aType

            model_lux = transform(model; preserve_ps_st=true)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

            @test model(x) ≈ model_lux(x, ps, st)[1]

            model_lux = transform(model)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

            @test size(model_lux(x, ps, st)[1]) == (2, 1)
        end

        @testset "Parallel" begin
            model = Flux.Parallel(+, Flux.Dense(2 => 2), Flux.Dense(2 => 2)) |>
                    fdevice(device)
            x = rand(Float32, 2, 1) |> aType

            model_lux = transform(model; preserve_ps_st=true)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

            @test model(x) ≈ model_lux(x, ps, st)[1]

            model_lux = transform(model)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

            @test size(model_lux(x, ps, st)[1]) == (2, 1)
        end

        @testset "Pairwise Fusion" begin
            model = Flux.PairwiseFusion(+, Flux.Dense(2 => 2), Flux.Dense(2 => 2)) |>
                    fdevice(device)
            x = (rand(Float32, 2, 1), rand(Float32, 2, 1)) .|> aType

            model_lux = transform(model; preserve_ps_st=true)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

            @test all(model(x) .≈ model_lux(x, ps, st)[1])

            model_lux = transform(model)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

            @test all(size.(model_lux(x, ps, st)[1]) .== ((2, 1),))
        end
    end

    @testset "Linear" begin
        @testset "Dense" begin
            for model in [
                Flux.Dense(2 => 4) |> fdevice(device),
                Flux.Dense(2 => 4; bias=false) |> fdevice(device),
            ]
                x = randn(Float32, 2, 4) |> aType

                model_lux = transform(model; preserve_ps_st=true)
                ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

                @test model(x) ≈ model_lux(x, ps, st)[1]

                model_lux = transform(model)
                ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

                @test size(model_lux(x, ps, st)[1]) == size(model(x))
            end
        end

        @testset "Scale" begin
            for model in [
                Flux.Scale(2) |> fdevice(device),
                Flux.Scale(2; bias=false) |> fdevice(device),
            ]
                x = randn(Float32, 2, 4) |> aType

                model_lux = transform(model; preserve_ps_st=true)
                ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

                @test model(x) ≈ model_lux(x, ps, st)[1]

                model_lux = transform(model)
                ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

                @test size(model_lux(x, ps, st)[1]) == size(model(x))
            end
        end

        @testset "Bilinear" begin
            for model in [
                Flux.Bilinear((2, 3) => 5) |> fdevice(device),
                Flux.Bilinear((2, 3) => 5; bias=false) |> fdevice(device),
            ]
                x = randn(Float32, 2, 4) |> aType
                y = randn(Float32, 3, 4) |> aType

                model_lux = transform(model; preserve_ps_st=true)
                ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

                @test model(x, y) ≈ model_lux((x, y), ps, st)[1]

                model_lux = transform(model)
                ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

                @test size(model_lux((x, y), ps, st)[1]) == size(model(x, y))
            end
        end

        @testset "Embedding" begin
            model = Flux.Embedding(16 => 4) |> fdevice(device)
            x = rand(1:16, 2, 4) |> aType

            model_lux = transform(model; preserve_ps_st=true)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

            @test model(x) ≈ model_lux(x, ps, st)[1]

            model_lux = transform(model)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

            @test size(model_lux(x, ps, st)[1]) == (4, 2, 4)
        end
    end

    @testset "Convolutions" begin
        @testset "Conv" begin
            model = Flux.Conv((3, 3), 1 => 2) |> fdevice(device)
            x = rand(Float32, 6, 6, 1, 4) |> aType

            model_lux = transform(model; preserve_ps_st=true)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

            @test model(x) ≈ model_lux(x, ps, st)[1]

            model = Flux.Conv((3, 3), 1 => 2; pad=Flux.SamePad()) |> fdevice(device)
            x = rand(Float32, 6, 6, 1, 4) |> aType

            model_lux = transform(model; preserve_ps_st=true)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

            @test model(x) ≈ model_lux(x, ps, st)[1]
        end

        @testset "CrossCor" begin
            model = Flux.CrossCor((3, 3), 1 => 2) |> fdevice(device)
            x = rand(Float32, 6, 6, 1, 4) |> aType

            model_lux = transform(model; preserve_ps_st=true)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

            @test model(x) ≈ model_lux(x, ps, st)[1]

            model = Flux.CrossCor((3, 3), 1 => 2; pad=Flux.SamePad()) |> fdevice(device)
            x = rand(Float32, 6, 6, 1, 4) |> aType

            model_lux = transform(model; preserve_ps_st=true)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

            @test model(x) ≈ model_lux(x, ps, st)[1]
        end

        @testset "ConvTranspose" begin
            model = Flux.ConvTranspose((3, 3), 1 => 2) |> fdevice(device)
            x = rand(Float32, 6, 6, 1, 4) |> aType

            model_lux = transform(model; preserve_ps_st=true)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

            @test model(x) ≈ model_lux(x, ps, st)[1]

            model = Flux.ConvTranspose((3, 3), 1 => 2; pad=Flux.SamePad()) |>
                    fdevice(device)
            x = rand(Float32, 6, 6, 1, 4) |> aType

            model_lux = transform(model; preserve_ps_st=true)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

            @test model(x) ≈ model_lux(x, ps, st)[1]
        end
    end

    @testset "Pooling" begin
        @testset "AdaptiveMaxPooling" begin
            model = Flux.AdaptiveMaxPool((2, 2)) |> fdevice(device)
            x = rand(Float32, 6, 6, 1, 4) |> aType

            model_lux = transform(model; preserve_ps_st=true)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

            @test model(x) ≈ model_lux(x, ps, st)[1]
        end

        @testset "AdaptiveMeanPooling" begin
            model = Flux.AdaptiveMeanPool((2, 2)) |> fdevice(device)
            x = rand(Float32, 6, 6, 1, 4) |> aType

            model_lux = transform(model; preserve_ps_st=true)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

            @test model(x) ≈ model_lux(x, ps, st)[1]
        end

        @testset "MaxPooling" begin
            model = Flux.MaxPool((2, 2)) |> fdevice(device)
            x = rand(Float32, 6, 6, 1, 4) |> aType

            model_lux = transform(model; preserve_ps_st=true)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

            @test model(x) ≈ model_lux(x, ps, st)[1]
        end

        @testset "MeanPooling" begin
            model = Flux.MeanPool((2, 2)) |> fdevice(device)
            x = rand(Float32, 6, 6, 1, 4) |> aType

            model_lux = transform(model; preserve_ps_st=true)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

            @test model(x) ≈ model_lux(x, ps, st)[1]
        end

        @testset "GlobalMaxPooling" begin
            model = Flux.GlobalMaxPool() |> fdevice(device)
            x = rand(Float32, 6, 6, 1, 4) |> aType

            model_lux = transform(model; preserve_ps_st=true)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

            @test model(x) ≈ model_lux(x, ps, st)[1]
        end

        @testset "GlobalMeanPooling" begin
            model = Flux.GlobalMeanPool() |> fdevice(device)
            x = rand(Float32, 6, 6, 1, 4) |> aType

            model_lux = transform(model; preserve_ps_st=true)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

            @test model(x) ≈ model_lux(x, ps, st)[1]
        end
    end

    @testset "Upsampling" begin
        @testset "Upsample" begin
            model = Flux.Upsample(5) |> fdevice(device)
            x = rand(Float32, 2, 2, 2, 1) |> aType

            model_lux = transform(model)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

            @test size(model_lux(x, ps, st)[1]) == (10, 10, 2, 1)
            @test model(x) ≈ model_lux(x, ps, st)[1]
        end

        @testset "PixelShuffle" begin
            model = Flux.PixelShuffle(2) |> fdevice(device)
            x = randn(Float32, 2, 2, 4, 1) |> aType

            model_lux = transform(model)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

            @test size(model_lux(x, ps, st)[1]) == (4, 4, 1, 1)
            @test model(x) ≈ model_lux(x, ps, st)[1]
        end
    end

    @testset "Recurrent" begin
        # @test_throws Lux.FluxModelConversionError transform(Flux.RNN(2 => 2))

        @testset "RNNCell" begin
            model = Flux.RNNCell(2 => 3) |> fdevice(device)
            x = rand(Float32, 2, 4) |> aType

            model_lux = transform(model)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

            @test size(model_lux(x, ps, st)[1][1]) == (3, 4)
        end

        @testset "LSTMCell" begin
            model = Flux.LSTMCell(2 => 3) |> fdevice(device)
            x = rand(Float32, 2, 4) |> aType

            model_lux = transform(model)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

            @test size(model_lux(x, ps, st)[1][1]) == (3, 4)
        end

        @testset "GRUCell" begin
            model = Flux.GRUCell(2 => 3) |> fdevice(device)
            x = rand(Float32, 2, 4) |> aType

            model_lux = transform(model)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

            @test size(model_lux(x, ps, st)[1][1]) == (3, 4)
        end
    end

    @testset "Normalize" begin
        @testset "BatchNorm" begin
            model = Flux.BatchNorm(2) |> fdevice(device)
            x = randn(Float32, 2, 4) |> aType

            model_lux = transform(model; preserve_ps_st=true)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device
            st = Lux.testmode(st)

            @test model(x) ≈ model_lux(x, ps, st)[1]

            x = randn(Float32, 2, 2, 2, 1) |> aType

            @test model(x) ≈ model_lux(x, ps, st)[1]

            model_lux = transform(model; preserve_ps_st=true, force_preserve=true)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device
            st = Lux.testmode(st)

            @test model(x) ≈ model_lux(x, ps, st)[1]
        end

        @testset "GroupNorm" begin
            model = Flux.GroupNorm(4, 2) |> fdevice(device)
            x = randn(Float32, 2, 2, 4, 1) |> aType

            model_lux = transform(model; preserve_ps_st=true)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device
            st = Lux.testmode(st)

            @test model(x) ≈ model_lux(x, ps, st)[1]

            model_lux = transform(model; preserve_ps_st=true, force_preserve=true)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device
            st = Lux.testmode(st)

            @test model(x) ≈ model_lux(x, ps, st)[1]
        end

        @testset "LayerNorm" begin
            model = Flux.LayerNorm(4) |> fdevice(device)
            x = randn(Float32, 4, 4, 4, 1) |> aType

            model_lux = transform(model; preserve_ps_st=true)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device
            st = Lux.testmode(st)

            @test model(x) ≈ model_lux(x, ps, st)[1]
        end

        @testset "InstanceNorm" begin
            model = Flux.InstanceNorm(4) |> fdevice(device)
            x = randn(Float32, 4, 4, 4, 1) |> aType

            model_lux = transform(model; preserve_ps_st=true)
            ps, st = Lux.setup(get_stable_rng(12345), model_lux) .|> device

            @test model(x) ≈ model_lux(x, ps, st)[1]
        end
    end

    @testset "Dropout" begin
        @testset "Dropout" begin
            model = transform(Flux.Dropout(0.5f0))

            x = randn(Float32, 2, 4) |> aType
            ps, st = Lux.setup(get_stable_rng(12345), model) .|> device

            @test size(model(x, ps, st)[1]) == size(x)

            x = randn(Float32, 2, 3, 4) |> aType
            ps, st = Lux.setup(get_stable_rng(12345), model) .|> device

            @test size(model(x, ps, st)[1]) == size(x)
        end

        @testset "AlphaDropout" begin
            model = transform(Flux.AlphaDropout(0.5))

            x = randn(Float32, 2, 4) |> aType
            ps, st = Lux.setup(get_stable_rng(12345), model) .|> device

            @test size(model(x, ps, st)[1]) == size(x)

            x = randn(Float32, 2, 4, 3) |> aType
            ps, st = Lux.setup(get_stable_rng(12345), model) .|> device

            @test size(model(x, ps, st)[1]) == size(x)
        end
    end

    @testset "Custom Layer" begin
        struct CustomFluxLayer
            weight::Any
            bias::Any
        end

        Flux.@functor CustomFluxLayer

        (c::CustomFluxLayer)(x) = c.weight .* x .+ c.bias

        c = CustomFluxLayer(randn(10), randn(10)) |> fdevice(device)
        x = randn(10) |> aType

        c_lux = transform(c)
        ps, st = Lux.setup(get_stable_rng(12345), c_lux) .|> device

        @test c(x) ≈ c_lux(x, ps, st)[1]
    end
end
