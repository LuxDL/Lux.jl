include("../shared_testsetup.jl")
include("../reactant_testsetup.jl")

using Reactant, Lux, Random
using LuxTestUtils: check_approx

@testset "Reactant: Alternate Precision" begin
    @testset "$(mode)" for (mode, atype, dev, ongpu) in MODES
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
