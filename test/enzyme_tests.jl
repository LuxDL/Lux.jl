# With LuxTestUtils 1.1 we have inbuilt Enzyme.jl support in `test_gradients`. We should be
# able to remove this, but this file is still helpful to catch errors in a localized way.
@testsetup module EnzymeTestSetup
using LuxTestUtils, Enzyme, Zygote, Test
using Lux, NNlib
using LuxTestUtils: check_approx

generic_loss_function(model, x, ps, st) = sum(abs2, first(model(x, ps, st)))

function compute_enzyme_gradient(model, x, ps, st)
    dx = Enzyme.make_zero(x)
    dps = Enzyme.make_zero(ps)
    Enzyme.autodiff(Reverse, generic_loss_function, Active, Const(model),
        Duplicated(x, dx), Duplicated(ps, dps), Const(st))
    return dx, dps
end

function compute_zygote_gradient(model, x, ps, st)
    _, dx, dps, _ = Zygote.gradient(generic_loss_function, model, x, ps, st)
    return dx, dps
end

function test_enzyme_gradients(model, x, ps, st)
    dx, dps = compute_enzyme_gradient(model, x, ps, st)
    dx_zygote, dps_zygote = compute_zygote_gradient(model, x, ps, st)
    @test check_approx(dx, dx_zygote; atol=1.0f-3, rtol=1.0f-3)
    @test check_approx(dps, dps_zygote; atol=1.0f-3, rtol=1.0f-3)
end

#! format: off
const MODELS_LIST  = [
    (Dense(2, 4), randn(Float32, 2, 3)),
    (Dense(2, 4, gelu), randn(Float32, 2, 3)),
    (Dense(2, 4, gelu; use_bias=false), randn(Float32, 2, 3)),
    (Chain(Dense(2, 4, relu), Dense(4, 3)), randn(Float32, 2, 3)),
    (Scale(2), randn(Float32, 2, 3)),
    (Conv((3, 3), 2 => 3), randn(Float32, 3, 3, 2, 2)),
    (Conv((3, 3), 2 => 3, gelu; pad=SamePad()), randn(Float32, 3, 3, 2, 2)),
    (Conv((3, 3), 2 => 3, relu; use_bias=false, pad=SamePad()), randn(Float32, 3, 3, 2, 2)),
    (Chain(Conv((3, 3), 2 => 3, gelu), Conv((3, 3), 3 => 1, gelu)), rand(Float32, 5, 5, 2, 2)),
    (Chain(Conv((4, 4), 2 => 2; pad=SamePad()), MeanPool((5, 5); pad=SamePad())), rand(Float32, 5, 5, 2, 2)),
    (Chain(Conv((3, 3), 2 => 3, relu; pad=SamePad()), MaxPool((2, 2))), rand(Float32, 5, 5, 2, 2)),
    (Maxout(() -> Dense(5 => 4, tanh), 3), randn(Float32, 5, 2)),
    (Bilinear((2, 2) => 3), randn(Float32, 2, 3)),
    (SkipConnection(Dense(2 => 2), vcat), randn(Float32, 2, 3)),
    (ConvTranspose((3, 3), 3 => 2; stride=2), rand(Float32, 5, 5, 3, 1)),
    (StatefulRecurrentCell(RNNCell(3 => 5)), rand(Float32, 3, 2)),
    (StatefulRecurrentCell(RNNCell(3 => 5, gelu)), rand(Float32, 3, 2)),
    (StatefulRecurrentCell(RNNCell(3 => 5, gelu; use_bias=false)), rand(Float32, 3, 2)),
    (Chain(StatefulRecurrentCell(RNNCell(3 => 5)), StatefulRecurrentCell(RNNCell(5 => 3))), rand(Float32, 3, 2)),
    (StatefulRecurrentCell(LSTMCell(3 => 5)), rand(Float32, 3, 2)),
    (Chain(StatefulRecurrentCell(LSTMCell(3 => 5)), StatefulRecurrentCell(LSTMCell(5 => 3))), rand(Float32, 3, 2)),
    (StatefulRecurrentCell(GRUCell(3 => 5)), rand(Float32, 3, 10)),
    (Chain(StatefulRecurrentCell(GRUCell(3 => 5)), StatefulRecurrentCell(GRUCell(5 => 3))), rand(Float32, 3, 10)),
    (Chain(Dense(2, 4), BatchNorm(4)), randn(Float32, 2, 3)),
    (Chain(Dense(2, 4), BatchNorm(4, gelu)), randn(Float32, 2, 3)),
    (Chain(Dense(2, 4), BatchNorm(4, gelu; track_stats=false)), randn(Float32, 2, 3)),
    (Chain(Conv((3, 3), 2 => 6), BatchNorm(6)), randn(Float32, 6, 6, 2, 2)),
    (Chain(Conv((3, 3), 2 => 6, tanh), BatchNorm(6)), randn(Float32, 6, 6, 2, 2)),
    (Chain(Dense(2, 4), GroupNorm(4, 2, gelu)), randn(Float32, 2, 3)),
    (Chain(Dense(2, 4), GroupNorm(4, 2)), randn(Float32, 2, 3)),
    (Chain(Conv((3, 3), 2 => 6), GroupNorm(6, 3)), randn(Float32, 6, 6, 2, 2)),
    (Chain(Conv((3, 3), 2 => 6, tanh), GroupNorm(6, 3)), randn(Float32, 6, 6, 2, 2)),
    (Chain(Conv((3, 3), 2 => 3, gelu), LayerNorm((1, 1, 3))), randn(Float32, 4, 4, 2, 2)),
    (Chain(Conv((3, 3), 2 => 6), InstanceNorm(6)), randn(Float32, 6, 6, 2, 2)),
    (Chain(Conv((3, 3), 2 => 6, tanh), InstanceNorm(6)), randn(Float32, 6, 6, 2, 2)),
]
#! format: on

export generic_loss_function, compute_enzyme_gradient, compute_zygote_gradient,
       test_enzyme_gradients, MODELS_LIST
end

@testitem "Enzyme Integration" setup=[EnzymeTestSetup, SharedTestSetup] tags=[
    :autodiff, :enzyme] begin
    rng=StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        # TODO: Currently all the tests are run on CPU. We should eventually add tests for
        #       CUDA and AMDGPU.
        ongpu && continue

        @testset "[$(i)] $(nameof(typeof(model)))" for (i, (model, x)) in
                                                       enumerate(MODELS_LIST)
            ps, st = Lux.setup(rng, model) |> dev
            x = x |> aType

            if LuxTestUtils.ENZYME_TESTING_ENABLED
                test_enzyme_gradients(model, x, ps, st)
            else
                @test_broken false
            end
        end
    end
end

@testitem "Enzyme Integration ComponentArray" setup=[EnzymeTestSetup, SharedTestSetup] tags=[
    :autodiff, :enzyme] begin
    using ComponentArrays

    rng=StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        # TODO: Currently all the tests are run on CPU. We should eventually add tests for
        #       CUDA and AMDGPU.
        ongpu && continue

        @testset "[$(i)] $(nameof(typeof(model)))" for (i, (model, x)) in
                                                       enumerate(MODELS_LIST)
            ps, st = Lux.setup(rng, model)
            ps = ComponentArray(ps)
            st = st |> dev
            x = x |> aType

            if LuxTestUtils.ENZYME_TESTING_ENABLED
                test_enzyme_gradients(model, x, ps, st)
            else
                @test_broken false
            end
        end
    end
end
