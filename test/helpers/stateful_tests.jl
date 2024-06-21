@testitem "Simple Stateful Tests" setup=[SharedTestSetup] tags=[:helpers] begin
    using Setfield

    rng = StableRNG(12345)

    struct NotFixedStateModel <: Lux.AbstractExplicitLayer end

    (m::NotFixedStateModel)(x, ps, st) = (x, (; s=1))

    model = NotFixedStateModel()
    ps, st = Lux.setup(rng, model)

    @test st isa NamedTuple{()}

    @test_deprecated StatefulLuxLayer(model, ps, st)

    smodel = StatefulLuxLayer{false}(model, ps, st)
    display(smodel)
    @test_nowarn smodel(1)

    smodel = StatefulLuxLayer{true}(model, ps, st)
    display(smodel)
    @test_throws ArgumentError smodel(1)

    @testset "Functors testing" begin
        model = Dense(2 => 3)
        ps, st = Lux.setup(rng, model)
        smodel = StatefulLuxLayer{true}(model, ps, st)

        @test Lux.parameterlength(smodel) == Lux.parameterlength(model)
        @test Lux.statelength(smodel) == Lux.statelength(model)

        x = Float32.(randn(rng, 2, 5))
        @test smodel(x) isa Matrix{Float32}

        smodel_f64 = f64(smodel)
        @test smodel_f64(x) isa Matrix{Float64}

        smodel_f64_2 = @set smodel_f64.ps = ps
        @test smodel_f64_2(x) isa Matrix{Float32}
    end
end
