using ADTypes, Optimisers, Tracker, ReverseDiff, Mooncake, ComponentArrays, Enzyme

include("../../shared_testsetup.jl")

@testset "AbstractADTypes" begin
    function _loss_function(model, ps, st, data)
        y, st = model(data, ps, st)
        return sum(y), st, ()
    end

    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        model = Dense(3, 2)
        opt = Adam(0.01f0)
        ps, st = dev(Lux.setup(rng, model))

        tstate = Training.TrainState(model, ps, st, opt)

        x = aType(randn(Lux.replicate(rng), Float32, (3, 1)))

        for ad in (AutoZygote(), AutoTracker(), AutoReverseDiff(), AutoEnzyme())
            ongpu && (ad isa AutoReverseDiff || ad isa AutoEnzyme) && continue
            !LuxTestUtils.ENZYME_TESTING_ENABLED[] && ad isa AutoEnzyme && continue
            !LuxTestUtils.ZYGOTE_TESTING_ENABLED[] && ad isa AutoZygote && continue

            grads, _, _, _ = Training.compute_gradients(ad, _loss_function, x, tstate)
            tstate_ = Training.apply_gradients(tstate, grads)
            @test tstate_.step == 1
            @test tstate != tstate_
        end
    end
end
