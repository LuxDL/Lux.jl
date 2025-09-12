@testitem "MultiHeadAttention" setup = [SharedTestSetup] tags = [:core_layers] begin
    using LinearAlgebra, NNlib

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

@testitem "Reactant MultiHeadAttention" setup = [
    SharedTestSetup, SharedReactantLayersTestSetup
] tags = [:reactant] begin
    using Reactant, Lux, NNlib
    using LuxTestUtils: check_approx

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
        cdev = cpu_device()

        mha = MultiHeadAttention(4 => 10 => 5; nheads=5)

        q = rand(Float32, (4, 3, 5))
        q_ra = dev(q)
        k = rand(Float32, (4, 3, 5))
        k_ra = dev(k)
        v = rand(Float32, (4, 3, 5))
        v_ra = dev(v)

        ps, st = Lux.setup(rng, mha)
        ps_ra, st_ra = dev((ps, st))

        (y_ra, α_ra), stₙ_ra = @jit mha((q_ra, k_ra, v_ra), ps_ra, st_ra)
        (y, α), stₙ = mha((q, k, v), ps, st)

        @test Array(y_ra) ≈ y atol = 1.0e-2 rtol = 1.0e-2
        @test Array(α_ra) ≈ α atol = 1.0e-2 rtol = 1.0e-2

        @testset "gradient" begin
            (∂q, ∂k, ∂v), ∂ps = ∇sumabs2_zygote(mha, (q, k, v), ps, st)
            (∂q_ra, ∂k_ra, ∂v_ra), ∂ps_ra = @jit ∇sumabs2_enzyme(
                mha, (q_ra, k_ra, v_ra), ps_ra, st_ra
            )
            @test ∂q_ra ≈ ∂q atol = 1.0e-2 rtol = 1.0e-2
            @test ∂k_ra ≈ ∂k atol = 1.0e-2 rtol = 1.0e-2
            @test ∂v_ra ≈ ∂v atol = 1.0e-2 rtol = 1.0e-2
            @test check_approx(∂ps_ra, ∂ps; atol=1.0e-2, rtol=1.0e-2)
        end
    end
end
