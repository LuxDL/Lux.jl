@testsetup module InstanceNormSetup
using LuxLib, LuxTestUtils, Random, Test, Zygote, NNlib

__is_training(::Val{training}) where {training} = training

function _setup_instancenorm(gen_f, aType, T, sz; affine::Bool=true)
    x = gen_f(T, sz) |> aType
    scale = affine ? aType(gen_f(T, sz[end - 1])) : nothing
    bias = affine ? aType(gen_f(T, sz[end - 1])) : nothing
    return x, scale, bias
end

anonact = x -> x^3

function run_instancenorm_testing(gen_f, T, sz, training, act, aType, mode, ongpu)
    _f = (args...) -> first(instancenorm(args..., training, act, epsilon))

    epsilon = LuxLib.__default_epsilon(T)
    x, scale, bias = _setup_instancenorm(gen_f, aType, T, sz)
    y, nt = instancenorm(x, scale, bias, training, act, epsilon)

    y_simple, nt_simple = instancenorm(x, scale, bias, training, act, epsilon)

    fp16 = T == Float16
    atol = fp16 ? 1.0f-2 : 1.0f-3
    rtol = fp16 ? 1.0f-2 : 1.0f-3

    @test y≈y_simple atol=atol rtol=rtol

    # Check the rrules
    if !fp16
        ∂x, ∂scale, ∂bias = Zygote.gradient(sum ∘ _f, x, scale, bias)
        ∂x_simple, ∂scale_simple, ∂bias_simple = Zygote.gradient(sum ∘ _f, x, scale, bias)
        @test ∂x≈∂x_simple atol=atol rtol=rtol
        @test ∂scale≈∂scale_simple atol=atol rtol=rtol
        @test ∂bias≈∂bias_simple atol=atol rtol=rtol
    end

    @test @inferred(instancenorm(x, scale, bias, training, act, epsilon)) isa Any
    @jet instancenorm(x, scale, bias, training, act, epsilon)

    if anonact !== act && __is_training(training)
        lfn = (x, sc, b, act, ϵ) -> sum(first(instancenorm(x, sc, b, Val(true), act, ϵ)))
        @test @inferred(Zygote.gradient(lfn, x, scale, bias, act, epsilon)) isa Any
    end

    @test y isa aType{T, length(sz)}
    @test size(y) == sz

    if __is_training(training) && !fp16
        __f = (args...) -> sum(first(instancenorm(args..., training, act, epsilon)))
        test_gradients(__f, x, scale, bias; atol, rtol, skip_backends=[AutoFiniteDiff()])
    end
end

const ALL_TEST_CONFIGS = Iterators.product(
    [Float16, Float32, Float64], ((4, 4, 6, 2), (3, 4, 2), (4, 4, 4, 3, 2)),
    (Val(true), Val(false)), (identity, relu, tanh_fast, sigmoid_fast, anonact))

const TEST_BLOCKS = collect(Iterators.partition(
    ALL_TEST_CONFIGS, ceil(Int, length(ALL_TEST_CONFIGS) / 5)))

export _setup_instancenorm, ALL_TEST_CONFIGS, TEST_BLOCKS, run_instancenorm_testing

end

@testitem "Instance Norm: Group 1" tags=[:instance_norm] setup=[
    SharedTestSetup, InstanceNormSetup] begin
    @testset "$mode" for (mode, aType, ongpu) in MODES
        @testset "eltype $T, size $sz, $training $act" for (T, sz, training, act) in TEST_BLOCKS[1]
            run_instancenorm_testing(
                __generate_fixed_array, T, sz, training, act, aType, mode, ongpu)
        end
    end
end

@testitem "Instance Norm: Group 2" tags=[:instance_norm] setup=[
    SharedTestSetup, InstanceNormSetup] begin
    @testset "$mode" for (mode, aType, ongpu) in MODES
        @testset "eltype $T, size $sz, $training $act" for (T, sz, training, act) in TEST_BLOCKS[2]
            run_instancenorm_testing(
                __generate_fixed_array, T, sz, training, act, aType, mode, ongpu)
        end
    end
end

@testitem "Instance Norm: Group 3" tags=[:instance_norm] setup=[
    SharedTestSetup, InstanceNormSetup] begin
    @testset "$mode" for (mode, aType, ongpu) in MODES
        @testset "eltype $T, size $sz, $training $act" for (T, sz, training, act) in TEST_BLOCKS[3]
            run_instancenorm_testing(
                __generate_fixed_array, T, sz, training, act, aType, mode, ongpu)
        end
    end
end

@testitem "Instance Norm: Group 4" tags=[:instance_norm] setup=[
    SharedTestSetup, InstanceNormSetup] begin
    @testset "$mode" for (mode, aType, ongpu) in MODES
        @testset "eltype $T, size $sz, $training $act" for (T, sz, training, act) in TEST_BLOCKS[4]
            run_instancenorm_testing(
                __generate_fixed_array, T, sz, training, act, aType, mode, ongpu)
        end
    end
end

@testitem "Instance Norm: Group 5" tags=[:instance_norm] setup=[
    SharedTestSetup, InstanceNormSetup] begin
    @testset "$mode" for (mode, aType, ongpu) in MODES
        @testset "eltype $T, size $sz, $training $act" for (T, sz, training, act) in TEST_BLOCKS[5]
            run_instancenorm_testing(
                __generate_fixed_array, T, sz, training, act, aType, mode, ongpu)
        end
    end
end
