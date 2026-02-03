include("../shared_testsetup.jl")
include("../reactant_testsetup.jl")

using Reactant, Lux, Random
using LuxTestUtils: check_approx

@testset "Recurrent Layers" begin
    # StableRNG uses UInt128 for seed that is not supported by Reactant inside loops
    rng = Random.default_rng()

    @testset "$(mode)" for (mode, atype, dev, ongpu) in MODES
        if ongpu
            Reactant.set_default_backend("gpu")
        else
            Reactant.set_default_backend("cpu")
        end

        dev = reactant_device(; force=true)

        @testset for cell in (RNNCell, LSTMCell, GRUCell)
            @testset for ordering in (BatchLastIndex(), TimeLastIndex())
                model = Recurrence(cell(4 => 4); ordering)
                ps, st = Lux.setup(rng, model)
                ps_ra, st_ra = dev((ps, st))
                if ordering isa TimeLastIndex
                    x = rand(Float32, 4, 12, 16)
                else
                    x = rand(Float32, 4, 16, 12)
                end
                x_ra = dev(x)

                y_ra, _ = @jit model(x_ra, ps_ra, st_ra)
                y, _ = model(x, ps, st)

                @test y_ra ≈ y atol = 1.0e-2 rtol = 1.0e-2

                @testset "Efficient Codegen" begin
                    hlo = @code_hlo model(x_ra, ps_ra, st_ra)
                    @test contains(repr(hlo), "stablehlo.while")
                    # ensure dead args elimination is working for while loop
                    @test !contains(repr(hlo), "stablehlo.dynamic_update_slice")
                end

                @testset "gradient" begin
                    ∂x_fd, ∂ps_fd = ∇sumabs2_reactant_fd(model, x_ra, ps_ra, st_ra)
                    @testset for mincut in (true, false), checkpointing in (false,)
                        model_ = Recurrence(cell(4 => 4); ordering, mincut, checkpointing)
                        ∂x_ra, ∂ps_ra = ∇sumabs2_reactant(model_, x_ra, ps_ra, st_ra)
                        @test ∂x_ra ≈ ∂x_fd atol = 1.0e-2 rtol = 1.0e-2
                        @test check_approx(∂ps_ra, ∂ps_fd; atol=1.0e-2, rtol=1.0e-2)
                    end
                end

                model2 = Recurrence(cell(4 => 4); ordering, return_sequence=true)
                sequence_ra, st_ra = @jit model2(x_ra, ps_ra, st_ra)
                @test sequence_ra[end] ≈ y atol = 1.0e-2 rtol = 1.0e-2
                @test length(sequence_ra) == 16

                @testset "Efficient Codegen" begin
                    hlo = @code_hlo model2(x_ra, ps_ra, st_ra)
                    @test contains(repr(hlo), "stablehlo.while")
                    @test contains(repr(hlo), "stablehlo.dynamic_update_slice")
                end
            end
        end
    end
end
