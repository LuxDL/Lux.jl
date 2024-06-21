@testitem "Simple Stateful Tests" setup=[SharedTestSetup] tags=[:helpers] begin
    rng = StableRNG(12345)

    struct NotFixedStateModel <: Lux.AbstractExplicitLayer end

    (m::NotFixedStateModel)(x, ps, st) = (x, (; s=1))

    model = NotFixedStateModel()
    ps, st = Lux.setup(rng, model)

    @test st isa NamedTuple{()}

    smodel = StatefulLuxLayer{false}(model, ps, st)
    display(smodel)
    @test_nowarn smodel(1)

    smodel = StatefulLuxLayer{true}(model, ps, st)
    display(smodel)
    @test_throws ArgumentError smodel(1)
end
