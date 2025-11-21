@testitem "Recurrent Layers" tags = [:reactant] setup = [
    SharedTestSetup, SharedReactantLayersTestSetup
] begin
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

                @testset "Efficient Codegen" begin
                    hlo = @code_hlo model(x_ra, ps_ra, st_ra)
                    @test contains(repr(hlo), "stablehlo.while")
                    # ensure dead args elimination is working for while loop
                    @test !contains(repr(hlo), "stablehlo.dynamic_update_slice")
                end

                @testset "gradient" begin
                    ∂x, ∂ps = ∇sumabs2_zygote(model, x, ps, st)
                    @testset for mincut in (true, false), checkpointing in (false,)
                        model_ = Recurrence(cell(4 => 4); ordering, mincut, checkpointing)
                        ∂x_ra, ∂ps_ra = @jit ∇sumabs2_enzyme(model_, x_ra, ps_ra, st_ra)
                        @test ∂x_ra ≈ ∂x atol = 1.0e-2 rtol = 1.0e-2
                        @test check_approx(∂ps_ra, ∂ps; atol=1.0e-2, rtol=1.0e-2)
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

@testitem "Dropout Layers" tags = [:reactant] setup = [SharedTestSetup] begin
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

            @test st.rng isa Reactant.ReactantRNG

            hlo = @code_hlo model(x, ps, st)
            @test contains(repr(hlo), "stablehlo.rng_bit_generator")

            y, st2 = @jit model(x, ps, st)
            @test st2.rng isa Reactant.ReactantRNG
            @test st.rng.seed != st2.rng.seed
        end
    end
end

@testitem "BatchNorm Layer" tags = [:reactant] setup = [
    SharedTestSetup, SharedReactantLayersTestSetup
] begin
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

@testitem "Reactant: RMSNorm" tags = [:reactant] setup = [
    SharedTestSetup, SharedReactantLayersTestSetup
] begin
    using Reactant, Lux, Random

    rng = StableRNG(12345)

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

        model = RMSNorm((4, 5))
        ps, st = Lux.setup(rng, model)
        ps_ra, st_ra = dev((ps, st))

        x = randn(rng, Float32, 4, 5, 3)
        x_ra = dev(x)

        y_ra, st_ra = @jit model(x_ra, ps_ra, st_ra)
        y, st = model(x, ps, st)

        @test y_ra ≈ y atol = 1.0e-4 rtol = 1.0e-2

        hlo = @code_hlo model(x_ra, ps_ra, st_ra)
        @test contains(repr(hlo), "stablehlo.rsqrt")
    end
end

@testitem "Reactant: Alternate Precision" tags = [:reactant] setup = [
    SharedTestSetup, SharedReactantLayersTestSetup
] begin
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

        model = Dense(2 => 3, tanh)
        model2 = AlternatePrecision{Float64}(model)

        x = randn(Float32, 2, 4)
        ps, st = Lux.setup(Random.default_rng(), model2)

        x_ra, ps_ra, st_ra = dev((x, ps, st))

        hlo1 = @code_hlo model(x_ra, ps_ra, st_ra)
        hlo2 = @code_hlo model2(x_ra, ps_ra, st_ra)
        @test !contains(repr(hlo1), "stablehlo.convert")
        @test !contains(repr(hlo1), "f64")
        @test contains(repr(hlo2), "stablehlo.convert")
        @test contains(repr(hlo2), "f64")
    end
end
