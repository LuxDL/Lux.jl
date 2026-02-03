using ADTypes, Optimisers, Tracker, ReverseDiff, Mooncake, ComponentArrays, Enzyme

include("../../shared_testsetup.jl")

@testset "TrainState" begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        model = Dense(3, 2)
        opt = Adam(0.01f0)
        ps, st = dev(Lux.setup(Lux.replicate(rng), model))

        tstate = Training.TrainState(model, ps, st, opt)

        x = randn(Lux.replicate(rng), Float32, (3, 1))
        opt_st = Optimisers.setup(opt, tstate.parameters)

        @test check_approx(tstate.optimizer_state, opt_st)
        @test tstate.step == 0
    end
end
