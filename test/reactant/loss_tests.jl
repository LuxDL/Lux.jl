@testitem "Compiled Loss Functions" tags=[:reactant] setup=[SharedTestSetup] begin
    using Reactant, Lux

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

        @testset "xlogx & xlogy" begin
            x = rand(rng, 10)
            y = rand(rng, 10)
            x_ra = Reactant.to_rarray(x)
            y_ra = Reactant.to_rarray(y)

            fn1(x) = LuxOps.xlogx.(x)
            fn2(x, y) = LuxOps.xlogy.(x, y)

            fn1_compiled = @compile fn1(x_ra)
            @test fn1(x) ≈ fn1_compiled(x_ra)

            fn2_compiled = @compile fn2(x_ra, y_ra)
            @test fn2(x, y) ≈ fn2_compiled(x_ra, y_ra)
        end

        @testset "Regression Loss" begin end

        @testset "Classification Loss" begin end

        @testset "Other Losses" begin end
    end
end
