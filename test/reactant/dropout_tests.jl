include("../shared_testsetup.jl")
include("../reactant_testsetup.jl")

using Reactant, Lux, Random
using LuxTestUtils: check_approx

@testset "Dropout Layers" begin
    @testset "$(mode)" for (mode, atype, dev, ongpu) in MODES
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
