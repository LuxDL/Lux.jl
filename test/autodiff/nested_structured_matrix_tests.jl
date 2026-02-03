using StableRNGs, Lux, ForwardDiff, Zygote, ComponentArrays, Functors, ADTypes
using LuxTestUtils, Test
using DispatchDoctor: allow_unstable

include("../shared_testsetup.jl")

@testset "Nested AD: Structured Matrix LuxDL/Lux.jl#602" begin
    rng = StableRNG(1234)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset "Structured Matrix: Issue LuxDL/Lux.jl#602" begin
            model = @compact(; potential=Dense(5 => 5, gelu)) do x
                @return reshape(diag(only(Zygote.jacobian(potential, x))), size(x))
            end

            ps, st = dev(Lux.setup(rng, model))
            x = aType(randn(rng, Float32, 5, 5))

            __f = let model = model, st = st
                (x, ps) -> sum(abs2, first(model(x, ps, st)))
            end

            @test_gradients(
                __f, x, ps; atol=1.0f-3, rtol=1.0f-3, skip_backends=[AutoEnzyme()]
            )
        end
    end
end
