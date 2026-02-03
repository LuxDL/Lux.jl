include("../shared_testsetup.jl")
include("../reactant_testsetup.jl")

using Reactant, Lux, NNlib, Random
using LuxTestUtils: check_approx

@testset "Reactant MultiHeadAttention" begin
    rng = Random.default_rng()

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
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
            (∂q_fd, ∂k_fd, ∂v_fd), ∂ps_fd = ∇sumabs2_reactant_fd(
                mha, (q_ra, k_ra, v_ra), ps_ra, st_ra
            )
            (∂q_ra, ∂k_ra, ∂v_ra), ∂ps_ra = ∇sumabs2_reactant(
                mha, (q_ra, k_ra, v_ra), ps_ra, st_ra
            )
            @test ∂q_ra ≈ ∂q_fd atol = 1.0e-2 rtol = 1.0e-2
            @test ∂k_ra ≈ ∂k_fd atol = 1.0e-2 rtol = 1.0e-2
            @test ∂v_ra ≈ ∂v_fd atol = 1.0e-2 rtol = 1.0e-2
            @test check_approx(∂ps_ra, ∂ps_fd; atol=1.0e-2, rtol=1.0e-2)
        end
    end
end
