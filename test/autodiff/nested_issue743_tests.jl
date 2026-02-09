using StableRNGs, Lux, ForwardDiff, Zygote, ComponentArrays, Functors, ADTypes
using LuxTestUtils, Test
using DispatchDoctor: allow_unstable

include("../shared_testsetup.jl")

@testset "Nested AD: Issue #743 (eval + gradient)" begin
    function loss_function(model, ps, st, x)
        smodel = StatefulLuxLayer(model, ps, st)
        y_pred = smodel(x)
        dy_pred = only(Zygote.gradient(sum ∘ smodel, x))
        loss = sum(dy_pred .+ y_pred .^ 2 / 2)
        return loss
    end

    rng = StableRNG(1234)
    model = Chain(Dense(1 => 8, sigmoid), Dense(8 => 1))
    ps, st = Lux.setup(rng, model)
    x = randn(rng, Float32, 1, 12)

    __f = let model = model, st = st
        (x, ps) -> loss_function(model, ps, st, x)
    end

    @test_gradients(__f, x, ps; atol=1.0f-3, rtol=1.0f-3, skip_backends=[AutoEnzyme()])
end
