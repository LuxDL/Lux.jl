@testitem "Embedding" setup = [SharedTestSetup] tags = [:core_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset "Linear indices" begin
            vocab_size, embed_size = 10, 4
            layer = Embedding(vocab_size => embed_size)
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))

            @test size(ps.weight) == (embed_size, vocab_size)
            @test LuxCore.outputsize(layer, nothing, rng) == (4,)

            x = rand(1:vocab_size, 1)[1]
            y, st_ = layer(x, ps, st)
            @test size(layer(x, ps, st)[1]) == (embed_size,)
            @test y == ps.weight[:, x]
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

            x = aType(rand(1:vocab_size, 3))
            y, st_ = layer(x, ps, st)
            @test y isa aType{Float32}
            @test y == ps.weight[:, x]
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

            x = aType(rand(1:vocab_size, 3, 4))
            y, st_ = layer(x, ps, st)
            @test y isa aType{Float32,3}
            @test size(y) == (embed_size, 3, 4)
            @test y == reshape(ps.weight[:, vec(x)], embed_size, 3, 4)
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)
        end

        @testset "Cartesian indices" begin
            vocab_size, embed_size = (5, 2), 4
            layer = Embedding(vocab_size => embed_size)
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))

            @test size(ps.weight) == (embed_size, vocab_size...)

            @test LuxCore.outputsize(layer, nothing, rng) == (4,)

            x = (rand(1:vocab_size[1], 1)[1], rand(1:vocab_size[2], 1)[1])
            y, st_ = layer(x, ps, st)
            @test size(layer(x, ps, st)[1]) == (embed_size,)
            @test y == ps.weight[:, x...]
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

            x = aType.((rand(1:vocab_size[1], 3), rand(1:vocab_size[2], 3)))
            y, st_ = layer(x, ps, st)
            @test y isa aType{Float32}
            @test y == ps.weight[:, CartesianIndex.(x...)]
            @jet layer(x, ps, st)
            @test_gradients(
                sumabs2first,
                layer,
                x,
                ps,
                st;
                atol=1.0f-3,
                rtol=1.0f-3,
                broken_backends=[AutoEnzyme()]
            )

            x = aType.((rand(1:vocab_size[1], 3, 4), rand(1:vocab_size[2], 3, 4)))
            y, st_ = layer(x, ps, st)
            @test y isa aType{Float32,3}
            @test size(y) == (embed_size, 3, 4)
            @jet layer(x, ps, st)
            @test_gradients(
                sumabs2first,
                layer,
                x,
                ps,
                st;
                atol=1.0f-3,
                rtol=1.0f-3,
                broken_backends=[AutoEnzyme()]
            )

            x = aType.((rand(1:vocab_size[1], 3), rand(1:vocab_size[2], 4)))
            @test_throws DimensionMismatch layer(x, ps, st)

            x = aType.((rand(1:vocab_size[1], 3, 4), rand(1:vocab_size[2], 4, 5)))
            @test_throws DimensionMismatch layer(x, ps, st)

            x = ()
            @test_throws ArgumentError layer(x, ps, st)
        end
    end
end

@testitem "SinusoidalPositionalEmbedding" setup = [SharedTestSetup] tags = [:core_layers] begin
    using LinearAlgebra

    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        model = SinusoidalPositionalEmbedding(16; min_freq=0.01f0)
        x = aType(collect(Float32, 0:9))
        ps, st = dev(Lux.setup(rng, model))

        y, st = model(x, ps, st)
        @test hasfield(typeof(st), :sigmas)
        @test size(y) == (16, 10)

        y_cpu = Array(y)
        similarities = y_cpu' * y_cpu
        @test maximum(abs, diag(similarities) .- 1) ≤ 1.0e-5

        @test_gradients(sumabs2first, model, x, ps, st; atol=1.0f-3, rtol=1.0f-3)
    end
end

@testitem "Reactant: SinusoidalPositionalEmbedding" setup = [
    SharedTestSetup, SharedReactantLayersTestSetup
] tags = [:reactant] begin
    using Reactant, Lux
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

        model = SinusoidalPositionalEmbedding(16; min_freq=0.01f0)
        x = collect(Float32, 0:9)
        x_ra = dev(x)
        ps, st = Lux.setup(rng, model)
        ps_ra, st_ra = dev((ps, st))

        y, st_ra = @jit model(x_ra, ps_ra, st_ra)
        @test hasfield(typeof(st_ra), :sigmas)
        @test size(y) == (16, 10)

        y_cpu = Array(y)
        similarities = y_cpu' * y_cpu
        @test maximum(abs, diag(similarities) .- 1) ≤ 1.0e-5

        @testset "gradient" begin
            ∂x, ∂ps = ∇sumabs2_zygote(model, x, ps, st)
            ∂x_ra, ∂ps_ra = @jit ∇sumabs2_enzyme(model, x_ra, ps_ra, st_ra)
            @test ∂x_ra ≈ ∂x atol = 1.0e-2 rtol = 1.0e-2
            @test check_approx(∂ps_ra, ∂ps; atol=1.0e-2, rtol=1.0e-2)
        end
    end
end

# https://github.com/mashu/PositionalEmbeddings.jl/blob/1252cd50093b0154c27d9f93ad2a914098a23c15/test/runtests.jl#L55
@testitem "RotaryPositionalEmbedding" setup = [SharedTestSetup] tags = [:core_layers] begin
    using LinearAlgebra

    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        model = RotaryPositionalEmbedding(
            16; max_sequence_length=10, base=10000, low_memory_variant=false
        )
        model_lm = RotaryPositionalEmbedding(
            16; max_sequence_length=10, base=10000, low_memory_variant=true
        )

        ps, st = Lux.setup(rng, model)
        ps_lm, st_lm = Lux.setup(rng, model_lm)

        expected_cos = reshape(
            [
                1.0 0.540302 -0.416147 -0.989992 -0.653644 0.283662 0.96017 0.753902 -0.1455 -0.91113
                1.0 0.950415 0.806578 0.582754 0.301137 -0.0103423 -0.320796 -0.599437 -0.818632 -0.956644
                1.0 0.995004 0.980067 0.955337 0.921061 0.877583 0.825336 0.764842 0.696707 0.62161
                1.0 0.9995 0.998001 0.995503 0.992011 0.987526 0.982054 0.9756 0.96817 0.959773
                1.0 0.99995 0.9998 0.99955 0.9992 0.99875 0.998201 0.997551 0.996802 0.995953
                1.0 0.999995 0.99998 0.999955 0.99992 0.999875 0.99982 0.999755 0.99968 0.999595
                1.0 1.0 0.999998 0.999996 0.999992 0.999987 0.999982 0.999976 0.999968 0.99996
                1.0 1.0 1.0 1.0 0.999999 0.999999 0.999998 0.999998 0.999997 0.999996
            ],
            (8, 10, 1),
        )[
            1:8, :,
        ]

        expected_sin = reshape(
            [
                0.0 0.841471 0.909297 0.14112 -0.756802 -0.958924 -0.279415 0.656987 0.989358 0.412118
                0.0 0.310984 0.591127 0.812649 0.953581 0.999947 0.947148 0.800422 0.574318 0.291259
                0.0 0.0998334 0.198669 0.29552 0.389418 0.479426 0.564642 0.644218 0.717356 0.783327
                0.0 0.0316175 0.0632034 0.0947261 0.126154 0.157456 0.1886 0.219556 0.250292 0.280778
                0.0 0.00999983 0.0199987 0.0299955 0.0399893 0.0499792 0.059964 0.0699428 0.0799147 0.0898785
                0.0 0.00316227 0.00632451 0.00948669 0.0126488 0.0158107 0.0189725 0.0221341 0.0252955 0.0284567
                0.0 0.001 0.002 0.003 0.00399999 0.00499998 0.00599996 0.00699994 0.00799991 0.00899988
                0.0 0.000316228 0.000632456 0.000948683 0.00126491 0.00158114 0.00189737 0.00221359 0.00252982 0.00284605
            ],
            (8, 10, 1),
        )

        @test st.cos_cache[1:8, :] ≈ expected_cos
        @test st.sin_cache[1:8, :] ≈ expected_sin
        @test st_lm.sin_cache ≈ expected_sin
        @test st_lm.cos_cache ≈ expected_cos

        x = dev(reshape(collect(Float32, 1:320), 16, 10, 2))
        x2 = dev(reshape(x, 16, 1, 10, 2))
        ps, st = dev((ps, st))
        ps_lm, st_lm = dev((ps_lm, st_lm))
        expected_sin = dev(expected_sin)
        expected_cos = dev(expected_cos)

        neg_half_x = similar(x)
        neg_half_x[1:8, :, :] .= -x[9:16, :, :]
        neg_half_x[9:16, :, :] .= x[1:8, :, :]

        y_expected =
            vcat(expected_cos, expected_cos) .* x .+
            vcat(expected_sin, expected_sin) .* neg_half_x

        y, st = model(x2, ps, st)
        y = dropdims(y; dims=2)
        @test size(y) == (16, 10, 2)
        @test y ≈ y_expected atol = 1.0e-3 rtol = 1.0e-3

        @test_gradients(sumabs2first, model, x2, ps, st; atol=1.0f-3, rtol=1.0f-3)

        y_lm, st_lm = model_lm(x2, ps_lm, st_lm)
        y_lm = dropdims(y_lm; dims=2)
        @test size(y_lm) == (16, 10, 2)
        @test y_lm ≈ y_expected atol = 1.0e-3 rtol = 1.0e-3

        @test_gradients(sumabs2first, model_lm, x2, ps_lm, st_lm; atol=1.0f-3, rtol=1.0f-3)
    end
end

@testitem "Reactant: RotaryPositionalEmbedding" setup = [
    SharedTestSetup, SharedReactantLayersTestSetup
] tags = [:reactant] begin
    using Reactant, Lux
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

        @testset for low_memory_variant in (true, false)
            model = RotaryPositionalEmbedding(
                16; max_sequence_length=10, base=10000, low_memory_variant
            )
            ps, st = Lux.setup(rng, model)

            x = reshape(collect(Float32, 1:320), 16, 1, 10, 2)
            x_ra = dev(reshape(x, 16, 1, 10, 2))
            ps_ra, st_ra = dev((ps, st))

            y_ra, st_ra = @jit model(x_ra, ps_ra, st_ra)
            y, st = model(x, ps, st)
            @test hasfield(typeof(st_ra), :cos_cache)
            @test hasfield(typeof(st_ra), :sin_cache)
            @test size(y_ra) == (16, 1, 10, 2)

            @test Array(y_ra) ≈ y atol = 1.0e-2 rtol = 1.0e-2

            @testset "gradient" begin
                ∂x, ∂ps = ∇sumabs2_zygote(model, x, ps, st)
                ∂x_ra, ∂ps_ra = @jit ∇sumabs2_enzyme(model, x_ra, ps_ra, st_ra)
                @test ∂x_ra ≈ ∂x atol = 1.0e-2 rtol = 1.0e-2
                @test check_approx(∂ps_ra, ∂ps; atol=1.0e-2, rtol=1.0e-2)
            end
        end
    end
end
