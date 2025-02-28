@testitem "Embedding" setup=[SharedTestSetup] tags=[:core_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset "Linear indices" begin
            vocab_size, embed_size = 10, 4
            layer = Embedding(vocab_size => embed_size)
            display(layer)
            ps, st = Lux.setup(rng, layer) |> dev

            @test size(ps.weight) == (embed_size, vocab_size)
            @test LuxCore.outputsize(layer, nothing, rng) == (4,)

            x = rand(1:vocab_size, 1)[1]
            y, st_ = layer(x, ps, st)
            @test size(layer(x, ps, st)[1]) == (embed_size,)
            @test y == ps.weight[:, x]
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

            x = rand(1:vocab_size, 3) |> aType
            y, st_ = layer(x, ps, st)
            @test y isa aType{Float32}
            @test y == ps.weight[:, x]
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

            x = rand(1:vocab_size, 3, 4) |> aType
            y, st_ = layer(x, ps, st)
            @test y isa aType{Float32, 3}
            @test size(y) == (embed_size, 3, 4)
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)
        end

        @testset "Cartesian indices" begin
            vocab_size, embed_size = (5, 2), 4
            layer = Embedding(vocab_size => embed_size)
            display(layer)
            ps, st = Lux.setup(rng, layer) |> dev

            @test size(ps.weight) == (embed_size, vocab_size...)

            @test LuxCore.outputsize(layer, nothing, rng) == (4,)

            x = (rand(1:vocab_size[1], 1)[1], rand(1:vocab_size[2], 1)[1])
            y, st_ = layer(x, ps, st)
            @test size(layer(x, ps, st)[1]) == (embed_size,)
            @test y == ps.weight[:, x...]
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

            x = (rand(1:vocab_size[1], 3), rand(1:vocab_size[2], 3)) .|> aType
            y, st_ = layer(x, ps, st)
            @test y isa aType{Float32}
            @test y == ps.weight[:, CartesianIndex.(x...)]
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3,
                broken_backends=[AutoEnzyme()])

            x = (rand(1:vocab_size[1], 3, 4), rand(1:vocab_size[2], 3, 4)) .|> aType
            y, st_ = layer(x, ps, st)
            @test y isa aType{Float32, 3}
            @test size(y) == (embed_size, 3, 4)
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3,
                broken_backends=[AutoEnzyme()])

            x = (rand(1:vocab_size[1], 3), rand(1:vocab_size[2], 4)) .|> aType
            @test_throws DimensionMismatch layer(x, ps, st)

            x = (rand(1:vocab_size[1], 3, 4), rand(1:vocab_size[2], 4, 5)) .|> aType
            @test_throws DimensionMismatch layer(x, ps, st)

            x = ()
            @test_throws ArgumentError layer(x, ps, st)
        end
    end
end
