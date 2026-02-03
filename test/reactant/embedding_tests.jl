include("../shared_testsetup.jl")
include("../reactant_testsetup.jl")

using Reactant, Lux, Random, LinearAlgebra
using LuxTestUtils: check_approx

@testset "Reactant: SinusoidalPositionalEmbedding" begin
    rng = Random.default_rng()

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
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
            (∂x_fd, ∂ps_fd) = ∇sumabs2_reactant_fd(model, x_ra, ps_ra, st_ra)
            (∂x_ra, ∂ps_ra) = ∇sumabs2_reactant(model, x_ra, ps_ra, st_ra)
            @test ∂x_ra ≈ ∂x_fd atol = 1.0e-2 rtol = 1.0e-2
            @test check_approx(∂ps_ra, ∂ps_fd; atol=1.0e-2, rtol=1.0e-2)
        end
    end
end

@testset "Reactant: RotaryPositionalEmbedding" begin
    rng = Random.default_rng()

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
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
                (∂x_fd, ∂ps_fd) = ∇sumabs2_reactant_fd(model, x_ra, ps_ra, st_ra)
                (∂x_ra, ∂ps_ra) = ∇sumabs2_reactant(model, x_ra, ps_ra, st_ra)
                @test ∂x_ra ≈ ∂x_fd atol = 1.0e-2 rtol = 1.0e-2
                @test check_approx(∂ps_ra, ∂ps_fd; atol=1.0e-2, rtol=1.0e-2)
            end
        end
    end
end
