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

        @testset for cell in (RNNCell, LSTMCell, GRUCell)
            model = Recurrence(cell(4 => 4))
            ps, st = Lux.setup(rng, model)
            ps_ra, st_ra = (ps, st) |> Reactant.to_rarray
            x = rand(Float32, 4, 16, 12)
            x_ra = x |> Reactant.to_rarray

            y_ra, _ = @jit model(x_ra, ps_ra, st_ra)
            y, _ = model(x, ps, st)

            @test y_ra≈y atol=1e-3 rtol=1e-3

            @testset "gradient" begin
                ∂x, ∂ps = ∇sumabs2_zygote(model, x, ps, st)
                ∂x_ra, ∂ps_ra = @jit ∇sumabs2_enzyme(model, x_ra, ps_ra, st_ra)
                @test ∂x_ra≈∂x atol=1e-3 rtol=1e-3
                @test check_approx(∂ps_ra, ∂ps; atol=1e-3, rtol=1e-3)
            end
        end
    end
end
