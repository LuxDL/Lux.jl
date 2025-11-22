@testitem "AutoDiff APIs: JVP and VJP" tags = [:reactant] setup = [SharedTestSetup] begin
    using Reactant, Lux, Enzyme, Zygote, Random, ForwardDiff
    using LuxTestUtils: check_approx

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

        models = (
            Chain(
                Dense(2 => 4, tanh), Dense(4 => 2), BranchLayer(NoOpLayer(), NoOpLayer())
            ),
            Chain(
                Chain(
                    Conv((3, 3), 2 => 3, gelu; pad=SamePad()),
                    Conv((3, 3), 3 => 2, gelu; pad=SamePad()),
                ),
                FlattenLayer(),
                Dense(18 => 1),
            ),
        )
        xs = (rand(rng, Float32, 2, 3), rand(rng, Float32, 3, 3, 2, 4))
        us = (rand(rng, Float32, 2, 3), rand(rng, Float32, 3, 3, 2, 4))
        vs = (
            (rand(rng, Float32, 2, 3), rand(rng, Float32, 2, 3)), rand(rng, Float32, 1, 4)
        )

        @testset "[$i]" for (i, model) in enumerate(models)
            ps, st = Lux.setup(rng, model)
            x, u, v = xs[i], us[i], vs[i]
            ps_ra, st_ra, x_ra, u_ra, v_ra = dev((ps, st, x, u, v))

            smodel = StatefulLuxLayer(model, ps, st)
            smodel_ra = StatefulLuxLayer(model, ps_ra, st_ra)

            jvp = jacobian_vector_product(smodel, AutoForwardDiff(), x, u)
            jvp_ra = @jit jacobian_vector_product(smodel_ra, AutoEnzyme(), x_ra, u_ra)
            @test check_approx(jvp, jvp_ra; atol=1e-5, rtol=1e-5)

            vjp = vector_jacobian_product(smodel, AutoZygote(), x, v)
            vjp_ra = @jit vector_jacobian_product(smodel_ra, AutoEnzyme(), x_ra, v_ra)
            @test check_approx(vjp, vjp_ra; atol=1e-5, rtol=1e-5)
        end
    end
end

@testitem "AutoDiff APIs: Batched Jacobian" tags = [:reactant] setup = [SharedTestSetup] begin
    using Reactant, Lux, Zygote, Random, Enzyme

    rng = Random.default_rng()

    models = (
        Chain(
            Conv((3, 3), 2 => 4, gelu; pad=SamePad()),
            Conv((3, 3), 4 => 2, gelu; pad=SamePad()),
            FlattenLayer(),
            Dense(18 => 2),
        ),
        Chain(Dense(2, 4, gelu), Dense(4, 2)),
    )
    Xs = (randn(rng, Float32, 3, 3, 2, 4), randn(rng, Float32, 2, 4))

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

        @testset "$(size(X))" for (model, X) in zip(models, Xs)
            ps, st = Lux.setup(rng, model)
            X_ra = dev(X)

            smodel = StatefulLuxLayer(model, ps, st)
            smodel_ra = StatefulLuxLayer(model, dev(ps), dev(st))

            J = batched_jacobian(smodel, AutoZygote(), X)
            J_ra = @jit batched_jacobian(smodel_ra, AutoEnzyme(; mode=Enzyme.Reverse), X_ra)
            J_ra2 = @jit batched_jacobian(
                smodel_ra, AutoEnzyme(; mode=Enzyme.Forward), X_ra
            )
            @test J ≈ J_ra atol = 1.0e-3 rtol = 1.0e-3
            @test J ≈ J_ra2 atol = 1.0e-3 rtol = 1.0e-3
        end
    end
end
