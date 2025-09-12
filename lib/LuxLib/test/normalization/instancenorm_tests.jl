@testsetup module InstanceNormSetup
using LuxLib, LuxTestUtils, Random, Test, Zygote, NNlib

is_training(::Val{training}) where {training} = training

function setup_instancenorm(gen_f, aType, T, sz; affine::Bool=true)
    x = aType(gen_f(T, sz))
    scale = affine ? aType(gen_f(T, sz[end - 1])) : nothing
    bias = affine ? aType(gen_f(T, sz[end - 1])) : nothing
    return x, scale, bias
end

anonact = x -> x^3

sumabs2instancenorm(args...) = sum(abs2, first(instancenorm(args...)))

function run_instancenorm_testing(gen_f, T, sz, training, act, aType)
    epsilon = LuxLib.Utils.default_epsilon(T)
    x, scale, bias = setup_instancenorm(gen_f, aType, T, sz)

    # First test without running stats
    y, nt = instancenorm(x, scale, bias, training, act, epsilon)

    atol = 1.0f-2
    rtol = 1.0f-2

    @test @inferred(instancenorm(x, scale, bias, training, act, epsilon)) isa Any
    @jet instancenorm(x, scale, bias, training, act, epsilon)

    @test y isa aType{T,length(sz)}
    @test size(y) == sz

    if is_training(training)
        @test_gradients(
            sumabs2instancenorm,
            x,
            scale,
            bias,
            training,
            act,
            epsilon;
            atol,
            rtol,
            soft_fail=[AutoFiniteDiff()],
        )
    end

    # Now test with running stats
    rm = aType(rand(T, sz[end - 1]))
    rv = aType(abs2.(gen_f(T, sz[end - 1])))

    y, nt = instancenorm(x, scale, bias, rm, rv, training, act, T(0.1), epsilon)

    @test @inferred(
        instancenorm(x, scale, bias, rm, rv, training, act, T(0.1), epsilon)
    ) isa Any
    @jet instancenorm(x, scale, bias, rm, rv, training, act, T(0.1), epsilon)

    @test y isa aType{T,length(sz)}
    @test size(y) == sz

    if is_training(training)
        @test_gradients(
            sumabs2instancenorm,
            x,
            scale,
            bias,
            Constant(rm),
            Constant(rv),
            training,
            act,
            T(0.1),
            epsilon;
            atol,
            rtol,
            soft_fail=[AutoFiniteDiff()],
        )
    end
end

const ALL_TEST_CONFIGS = Iterators.product(
    [Float32, Float64],
    # ((4, 4, 6, 2), (3, 4, 2), (4, 4, 4, 3, 2)),
    ((4, 4, 6, 2),),
    (Val(true), Val(false)),
    (identity, sigmoid_fast, anonact),
)

const TEST_BLOCKS = collect(
    Iterators.partition(ALL_TEST_CONFIGS, ceil(Int, length(ALL_TEST_CONFIGS) / 2))
)

export setup_instancenorm, ALL_TEST_CONFIGS, TEST_BLOCKS, run_instancenorm_testing

end

@testitem "Instance Norm: Group 1" tags = [:normalization] setup = [
    SharedTestSetup, InstanceNormSetup
] begin
    @testset "$mode" for (mode, aType, ongpu, fp64) in MODES
        @testset "eltype $T, size $sz, $training $act" for (T, sz, training, act) in
                                                           TEST_BLOCKS[1]
            !fp64 && T == Float64 && continue
            run_instancenorm_testing(generate_fixed_array, T, sz, training, act, aType)
        end
    end
end

@testitem "Instance Norm: Group 2" tags = [:normalization] setup = [
    SharedTestSetup, InstanceNormSetup
] begin
    @testset "$mode" for (mode, aType, ongpu, fp64) in MODES
        @testset "eltype $T, size $sz, $training $act" for (T, sz, training, act) in
                                                           TEST_BLOCKS[2]
            !fp64 && T == Float64 && continue
            run_instancenorm_testing(generate_fixed_array, T, sz, training, act, aType)
        end
    end
end
