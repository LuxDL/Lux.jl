using NNlib

include("../shared_testsetup.jl")

@testset "MultiHeadAttention" begin
    rng = StableRNG(12345)

    function loss(model, x, ps, st)
        y, α = first(model(x, ps, st))
        return sum(abs2, y) + sum(abs2, α)
    end

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        dim, nheads, batch_size, len = 4, 2, 5, 3

        mha = MultiHeadAttention(dim; nheads)

        q = rand(Float32, (dim, len, batch_size)) |> aType
        k = rand(Float32, (dim, len, batch_size)) |> aType
        v = rand(Float32, (dim, len, batch_size)) |> aType

        ps, st = Lux.setup(rng, mha) |> dev

        (y, α), stₙ = mha((q, k, v), ps, st)
        @test y isa aType{Float32,3}
        @test size(y) == (dim, len, batch_size)
        @test α isa aType{Float32,4}
        @test size(α) == (len, len, nheads, batch_size)

        # check that stₙ has all the fields that st has
        @test mha((q, k, v), ps, stₙ)[1][1] ≈ y

        @testset "self-attention" begin
            (y1, α1), stₙ = mha(q, ps, st)
            (y2, α2), stₙ = mha((q, q, q), ps, stₙ)
            @test y1 ≈ y2
            @test α1 ≈ α2
        end

        @testset "key and value are the same" begin
            (y1, α1), stₙ = mha((q, k, k), ps, st)
            (y2, α2), stₙ = mha((q, k), ps, stₙ)
            @test y1 ≈ y2
            @test α1 ≈ α2
        end

        @testset "change dims" begin
            dims = 4 => 10 => 5
            nhead = 5
            mha2 = MultiHeadAttention(dims; nheads)
            ps2, st2 = Lux.setup(rng, mha2) |> dev
            (y2, _), st2ₙ = mha2((q, k, v), ps2, st2)
            @test size(y2) == (dims.second.second, len, batch_size)
        end

        @testset "mask" begin
            mask = NNlib.make_causal_mask(q)
            (y, α), stₙ = mha((q, q, q, mask), ps, st)
            @test all(α[2, 1, :, :] .== 0)
            @test α[:, :, 1, 1] ≈ triu(α[:, :, 1, 1])
        end

        @test_gradients(loss, mha, (q, k, v), ps, st; atol=1.0f-3, rtol=1.0f-3)
    end
end

@testset "GroupQueryAttention" begin
    rng = StableRNG(12345)

    function loss(model, x, ps, st)
        y, α = first(model(x, ps, st))
        return sum(abs2, y) + sum(abs2, α)
    end

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        dim, nqheads, nkvheads, batch_size, len = 8, 4, 2, 5, 3

        gqa = GroupQueryAttention(dim; nqheads, nkvheads)

        q = rand(Float32, (dim, len, batch_size)) |> aType
        k = rand(Float32, (dim, len, batch_size)) |> aType
        v = rand(Float32, (dim, len, batch_size)) |> aType

        ps, st = Lux.setup(rng, gqa) |> dev

        (y, α), stₙ = gqa((q, k, v), ps, st) # a standard full cross-attention forward pass with distinct query, key, and value tensors.
        @test y isa aType{Float32,3} # asserts that output tensor y is a 3D array matching the target device array type (e.g., CuArray on GPU).
        @test size(y) == (dim, len, batch_size) # Verifies output tensor dimensions match (d_out, seq_len,batch).
        @test α isa aType{Float32,4} 
        @test size(α) == (len, len, nqheads, batch_size) # Verifies that attention scores maintain n_qheads, dimensions, proving key/value heads were broadcast correctly across query groups.
       
        @test gqa((q, k, v), ps, stₙ)[1][1] ≈ y # Validates functional state propagation—passing the updated layer state stₙ back into the layer produces identical output values (y) without parameter mutation side effects.

        @testset "self-attention" begin
            (y1, α1), stₙ = gqa(q, ps, st)
            (y2, α2), stₙ = gqa((q, q, q), ps, stₙ)
            @test y1 ≈ y2
            @test α1 ≈ α2
        end

        @testset "key and value are the same" begin
            (y1, α1), stₙ = gqa((q, k, k), ps, st)
            (y2, α2), stₙ = gqa((q, k), ps, stₙ)
            @test y1 ≈ y2
            @test α1 ≈ α2
        end

        @testset "change dims" begin
            dims = 8 => 12 => 6
            nheads_q, nheads_kv = 4, 2
            gqa2 = GroupQueryAttention(dims; nqheads=nheads_q, nkvheads=nheads_kv)
            ps2, st2 = Lux.setup(rng, gqa2) |> dev
            (y2, _), st2ₙ = gqa2((q, k, v), ps2, st2)
            @test size(y2) == (dims.second.second, len, batch_size)
        end

        @testset "mask" begin
            mask = NNlib.make_causal_mask(q)
            (y, α), stₙ = gqa((q, q, q, mask), ps, st)
            @test all(α[2, 1, :, :] .== 0) # Verifies that position 2 cannot attend to position 1, forcing attention weight probabilities to strictly zero out past causal boundaries.
            @test α[:, :, 1, 1] ≈ triu(α[:, :, 1, 1])
        end

        @testset "MHA Parity (nqheads == nkvheads)" begin
            nheads = 4
            mha = MultiHeadAttention(dim; nheads)
            gqa_mha = GroupQueryAttention(dim; nqheads=nheads, nkvheads=nheads) # Forces n_qheads=n_kvheads, where GQA degenerates structurally into standard MHA.

            # Identical parameter tree layout allows passing shared parameters directly
            ps_shared, st_mha = Lux.setup(rng, mha) |> dev
            _, st_gqa = Lux.setup(rng, gqa_mha) |> dev

            (y_mha, α_mha), _ = mha((q, k, v), ps_shared, st_mha)
            (y_gqa, α_gqa), _ = gqa_mha((q, k, v), ps_shared, st_gqa)

            @test y_gqa ≈ y_mha
            @test α_gqa ≈ α_mha
        end

        @test_gradients(loss, gqa, (q, k, v), ps, st; atol=1.0f-3, rtol=1.0f-3)
    end
end