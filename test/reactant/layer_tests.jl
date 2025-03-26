@testitem "Recurrent Layers" tags = [:reactant] setup = [
    SharedTestSetup, SharedReactantLayersTestSetup
] skip = :(Sys.iswindows()) begin
    using Reactant, Lux
    using LuxTestUtils: check_approx

    # StableRNG uses UInt128 for seed that is not supported by Reactant inside loops
    rng = Random.default_rng()

    @testset "$(mode)" for (mode, atype, dev, ongpu) in MODES
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

                @testset "gradient" begin
                    ∂x, ∂ps = ∇sumabs2_zygote(model, x, ps, st)
                    ∂x_ra, ∂ps_ra = @jit ∇sumabs2_enzyme(model, x_ra, ps_ra, st_ra)
                    @test ∂x_ra ≈ ∂x atol = 1.0e-2 rtol = 1.0e-2
                    @test check_approx(∂ps_ra, ∂ps; atol=1.0e-2, rtol=1.0e-2)
                end
            end
        end
    end
end

@testitem "Dropout Layers" tags = [:reactant] setup = [SharedTestSetup] skip =
    :(Sys.iswindows()) begin
    using Reactant, Lux, Random

    @testset "$(mode)" for (mode, atype, dev, ongpu) in MODES
        if mode == "amdgpu"
            @warn "Skipping AMDGPU tests for Reactant"
            continue
        end

        dev = reactant_device(; force=true)

        if ongpu
            Reactant.set_default_backend("gpu")
        else
            Reactant.set_default_backend("cpu")
        end

        @testset for layer in (AlphaDropout, Dropout, VariationalHiddenDropout)
            model = layer(0.5f0)
            ps, st = dev(Lux.setup(Random.default_rng(), model))
            x = dev(randn(Float32, 10, 10))

            @test st.rng isa Reactant.ConcreteRNG

            hlo = @code_hlo model(x, ps, st)
            @test contains(repr(hlo), "stablehlo.rng_bit_generator")

            y, st2 = @jit model(x, ps, st)
            @test st2.rng isa Reactant.ConcreteRNG
            @test st.rng.seed != st2.rng.seed
        end
    end
end

@testitem "BatchNorm Layer" tags = [:reactant] setup = [
    SharedTestSetup, SharedReactantLayersTestSetup
] skip = :(Sys.iswindows()) begin
    using Reactant, Lux, Random

    @testset "$(mode)" for (mode, atype, dev, ongpu) in MODES
        if mode == "amdgpu"
            @warn "Skipping AMDGPU tests for Reactant"
            continue
        end

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
                ∂x, ∂ps = ∇sumabs2_zygote(model, x, ps, st)
                ∂x_ra, ∂ps_ra = @jit ∇sumabs2_enzyme(model, x_ra, ps_ra, st_ra)
                @test ∂x_ra ≈ ∂x atol = 1.0e-2 rtol = 1.0e-2
                @test check_approx(∂ps_ra, ∂ps; atol=1.0e-2, rtol=1.0e-2)
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
