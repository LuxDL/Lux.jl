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

@testitem "SinusoidalPositionalEmbedding" setup=[SharedTestSetup] tags=[:core_layers] begin
    using LinearAlgebra

    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        model = SinusoidalPositionalEmbedding(16; min_freq=0.01f0)
        x = aType(collect(Float32, 0:9))
        ps, st = Lux.setup(rng, model) |> dev

        y, st = model(x, ps, st)
        @test hasfield(typeof(st), :sigmas)
        @test size(y) == (16, 10)

        y_cpu = Array(y)
        similarities = y_cpu' * y_cpu
        @test maximum(abs, diag(similarities) .- 1) ≤ 1e-5

        @test_gradients(sumabs2first, model, x, ps, st; atol=1.0f-3, rtol=1.0f-3)
    end
end

@testitem "SinusoidalPositionalEmbedding" setup=[
    SharedTestSetup, SharedReactantLayersTestSetup] tags=[:reactant] begin
    using Reactant, Lux
    using LuxTestUtils: check_approx

    # StableRNG uses UInt128 for seed that is not supported by Reactant inside loops
    rng = Random.default_rng()

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        if mode == "amdgpu"
            @warn "Skipping AMDGPU tests for Reactant"
            continue
        end

        if ongpu
            Reactant.set_default_backend("gpu")
        else
            Reactant.set_default_backend("cpu")
        end

        dev = reactant_device(; force=true)

        model = SinusoidalPositionalEmbedding(16; min_freq=0.01f0)
        x = collect(Float32, 0:9)
        x_ra = x |> dev
        ps, st = Lux.setup(rng, model)
        ps_ra, st_ra = (ps, st) |> dev

        y, st = @jit model(x_ra, ps_ra, st_ra)
        @test hasfield(typeof(st_ra), :sigmas)
        @test size(y) == (16, 10)

        y_cpu = Array(y)
        similarities = y_cpu' * y_cpu
        @test maximum(abs, diag(similarities) .- 1) ≤ 1e-5

        @testset "gradient" begin
            ∂x, ∂ps = ∇sumabs2_zygote(model, x, ps, st)
            ∂x_ra, ∂ps_ra = @jit ∇sumabs2_enzyme(model, x_ra, ps_ra, st_ra)
            @test ∂x_ra≈∂x atol=1e-2 rtol=1e-2
            @test check_approx(∂ps_ra, ∂ps; atol=1e-2, rtol=1e-2)
        end
    end
end
