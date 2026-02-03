include("../shared_testsetup.jl")
include("../reactant_testsetup.jl")

using Reactant, Lux, Random
using LuxTestUtils: check_approx

@testset "Reactant: RMSNorm" begin
    rng = StableRNG(12345)

    @testset "$(mode)" for (mode, atype, dev, ongpu) in MODES
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
