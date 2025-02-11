@testsetup module SharedReactantLayersTestSetup

using Lux, Reactant, Enzyme, Zygote

sumabs2(model, x, ps, st) = sum(abs2, first(model(x, ps, st)))

function ∇sumabs2_zygote(model, x, ps, st)
    return Zygote.gradient((x, ps) -> sumabs2(model, x, ps, st), x, ps)
end

function ∇sumabs2_enzyme(model, x, ps, st)
    dx = Enzyme.make_zero(x)
    dps = Enzyme.make_zero(ps)
    Enzyme.autodiff(
        Enzyme.Reverse, sumabs2, Active,
        Const(model), Duplicated(x, dx),
        Duplicated(ps, dps), Const(st)
    )
    return dx, dps
end

export ∇sumabs2_zygote, ∇sumabs2_enzyme

end

@testitem "Recurrent Layers" tags=[:reactant] setup=[
    SharedTestSetup, SharedReactantLayersTestSetup] skip=:(Sys.iswindows()) begin
    using Reactant, Lux
    using LuxTestUtils: check_approx

    rng = StableRNG(123)

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
                ps_ra, st_ra = (ps, st) |> dev
                if ordering isa TimeLastIndex
                    x = rand(Float32, 4, 12, 16)
                else
                    x = rand(Float32, 4, 16, 12)
                end
                x_ra = x |> dev

                y_ra, _ = @jit model(x_ra, ps_ra, st_ra)
                y, _ = model(x, ps, st)

                @test y_ra≈y atol=1e-2 rtol=1e-2

                @testset "gradient" begin
                    ∂x, ∂ps = ∇sumabs2_zygote(model, x, ps, st)
                    ∂x_ra, ∂ps_ra = @jit ∇sumabs2_enzyme(model, x_ra, ps_ra, st_ra)
                    @test ∂x_ra≈∂x atol=1e-2 rtol=1e-2
                    @test check_approx(∂ps_ra, ∂ps; atol=1e-2, rtol=1e-2)
                end
            end
        end
    end
end

@testitem "Dropout Layers" tags=[:reactant] setup=[SharedTestSetup] skip=:(Sys.iswindows()) begin
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
            ps, st = Lux.setup(Random.default_rng(), model) |> dev
            x = randn(Float32, 10, 10) |> dev

            @test st.rng isa Reactant.ConcreteRNG

            hlo = @code_hlo model(x, ps, st)
            @test contains(repr(hlo), "stablehlo.rng_bit_generator")

            y, st2 = @jit model(x, ps, st)
            @test st2.rng isa Reactant.ConcreteRNG
            @test st.rng.seed != st2.rng.seed
        end
    end
end

@testitem "BatchNorm Layer" tags=[:reactant] setup=[
    SharedTestSetup, SharedReactantLayersTestSetup] skip=:(Sys.iswindows()) begin
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

        @testset for track_stats in (true, false), affine in (true, false),
            act in (identity, tanh)

            model = Chain(
                Dense(2 => 3, tanh),
                BatchNorm(3, act; track_stats, affine, init_bias=rand32, init_scale=rand32),
                Dense(3 => 2)
            )

            x = rand(Float32, 2, 4)
            ps, st = Lux.setup(Random.default_rng(), model)

            x_ra = x |> dev
            ps_ra = ps |> dev
            st_ra = st |> dev

            y, st2 = model(x, ps, st)
            y_ra, st2_ra = @jit model(x_ra, ps_ra, st_ra)

            @test y≈y_ra rtol=1e-3 atol=1e-3
            if track_stats
                @test st2.layer_2.running_mean≈st2_ra.layer_2.running_mean rtol=1e-3 atol=1e-3
                @test st2.layer_2.running_var≈st2_ra.layer_2.running_var rtol=1e-3 atol=1e-3
            end

            # TODO: Check for stablehlo.batch_norm_training once we emit it in LuxLib

            @testset "gradient" begin
                ∂x, ∂ps = ∇sumabs2_zygote(model, x, ps, st)
                ∂x_ra, ∂ps_ra = @jit ∇sumabs2_enzyme(model, x_ra, ps_ra, st_ra)
                @test ∂x_ra≈∂x atol=1e-2 rtol=1e-2
                @test check_approx(∂ps_ra, ∂ps; atol=1e-2, rtol=1e-2)
            end

            y2, st3 = model(x, ps, Lux.testmode(st2))
            y2_ra, st3_ra = @jit model(x_ra, ps_ra, Lux.testmode(st2_ra))

            @test y2≈y2_ra rtol=1e-3 atol=1e-3
            if track_stats
                @test st3.layer_2.running_mean≈st3_ra.layer_2.running_mean rtol=1e-3 atol=1e-3
                @test st3.layer_2.running_var≈st3_ra.layer_2.running_var rtol=1e-3 atol=1e-3
            end

            hlo = @code_hlo model(x_ra, ps_ra, Lux.testmode(st_ra))
            @test contains(repr(hlo), "stablehlo.batch_norm_inference")
        end
    end
end
