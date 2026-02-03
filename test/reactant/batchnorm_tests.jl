include("../shared_testsetup.jl")
include("../reactant_testsetup.jl")

using Reactant, Lux, Random
using LuxTestUtils: check_approx

@testset "BatchNorm Layer" begin
    @testset "$(mode)" for (mode, atype, dev, ongpu) in MODES
        dev = reactant_device(; force=true)

        if ongpu
            Reactant.set_default_backend("gpu")
        else
            Reactant.set_default_backend("cpu")
        end

        @testset for track_stats in (true, false),
            affine in (true, false),
            act in (identity, tanh)

            model = Chain(
                Dense(2 => 3, tanh),
                BatchNorm(3, act; track_stats, affine, init_bias=rand32, init_scale=rand32),
                Dense(3 => 2),
            )

            model_decomposed = Chain(
                Dense(2 => 3, tanh),
                BatchNorm(
                    3,
                    act;
                    track_stats,
                    affine,
                    init_bias=rand32,
                    init_scale=rand32,
                    use_decomposed_implementation=true,
                ),
                Dense(3 => 2),
            )

            x = rand(Float32, 2, 4)
            ps, st = Lux.setup(Random.default_rng(), model)

            x_ra = dev(x)
            ps_ra = dev(ps)
            st_ra = dev(st)

            y, st2 = model(x, ps, st)
            y_ra, st2_ra = @jit model(x_ra, ps_ra, st_ra)

            @test y ≈ y_ra rtol = 1.0e-3 atol = 1.0e-3
            if track_stats
                @test st2.layer_2.running_mean ≈ st2_ra.layer_2.running_mean rtol = 1.0e-3 atol =
                    1.0e-3
                @test st2.layer_2.running_var ≈ st2_ra.layer_2.running_var rtol = 1.0e-3 atol =
                    1.0e-3
            end

            @testset "gradient" begin
                # batching for native batchnorm ops is currently busted upstream
                # See https://github.com/EnzymeAD/Enzyme-JAX/issues/1947
                (∂x_fd, ∂ps_fd) = ∇sumabs2_reactant_fd(model_decomposed, x_ra, ps_ra, st_ra)
                (∂x_ra, ∂ps_ra) = ∇sumabs2_reactant(model, x_ra, ps_ra, st_ra)
                @test ∂x_ra ≈ ∂x_fd atol = 1.0e-2 rtol = 1.0e-2
                @test check_approx(∂ps_ra, ∂ps_fd; atol=1.0e-2, rtol=1.0e-2)
            end

            y2, st3 = model(x, ps, Lux.testmode(st2))
            y2_ra, st3_ra = @jit model(x_ra, ps_ra, Lux.testmode(st2_ra))

            @test y2 ≈ y2_ra rtol = 1.0e-3 atol = 1.0e-3
            if track_stats
                @test st3.layer_2.running_mean ≈ st3_ra.layer_2.running_mean rtol = 1.0e-3 atol =
                    1.0e-3
                @test st3.layer_2.running_var ≈ st3_ra.layer_2.running_var rtol = 1.0e-3 atol =
                    1.0e-3
            end
        end
    end
end
