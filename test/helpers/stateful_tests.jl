@testitem "Simple Stateful Tests" setup=[SharedTestSetup] tags=[:helpers] begin
    using Setfield, Zygote

    rng=StableRNG(12345)

    struct NotFixedStateModel<:Lux.AbstractLuxLayer end

    (m::NotFixedStateModel)(x, ps, st)=(x, (; s=1))

    model=NotFixedStateModel()
    ps, st=Lux.setup(rng, model)

    @test st isa NamedTuple{()}

    smodel=StatefulLuxLayer{false}(model, ps, st)
    display(smodel)
    @test smodel(1) isa Any

    smodel=StatefulLuxLayer{true}(model, ps, st)
    display(smodel)
    @test_throws ArgumentError smodel(1)

    @testset "Functors testing" begin
        model = Dense(2 => 3)
        ps, st = Lux.setup(rng, model)
        smodel = StatefulLuxLayer{true}(model, ps, st)

        @test Lux.parameterlength(smodel) == Lux.parameterlength(model)
        @test Lux.statelength(smodel) == Lux.statelength(model)

        smodel2 = StatefulLuxLayer{false}(model, ps, st)
        @test Lux.parameterlength(smodel2) == Lux.parameterlength(model)
        @test Lux.statelength(smodel2) == Lux.statelength(model)

        x = Float32.(randn(rng, 2, 5))
        @test smodel(x) isa Matrix{Float32}

        smodel_f64 = f64(smodel)
        @test smodel_f64(x) isa Matrix{Float64}

        smodel_f64_2 = @set smodel_f64.ps = ps
        @test smodel_f64_2(x) isa Matrix{Float32}

        smodel = StatefulLuxLayer{true}(model, ps, (; x=2))
        myloss(m) = m.st.x
        @test only(Zygote.gradient(myloss, smodel)) === nothing
    end

    @testset "Updating State" begin
        model = BatchNorm(3, relu)
        ps, st = Lux.setup(rng, model)

        smodel = StatefulLuxLayer{true}(model, ps, st)
        @test smodel.st.training isa Val{true}

        smodel = LuxCore.testmode(smodel)
        @test smodel.st.training isa Val{false}

        smodel = LuxCore.trainmode(smodel)
        @test smodel.st.training isa Val{true}

        smodel = LuxCore.update_state(smodel, :training, 2)
        @test smodel.st.training == 2

        smodel = StatefulLuxLayer{false}(model, ps, st)
        @test smodel.st_any.training isa Val{true}

        smodel = LuxCore.testmode(smodel)
        @test smodel.st_any.training isa Val{false}

        smodel = LuxCore.trainmode(smodel)
        @test smodel.st_any.training isa Val{true}

        smodel = LuxCore.update_state(smodel, :training, 2)
        @test smodel.st_any.training == 2
    end
end
