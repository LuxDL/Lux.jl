@testsetup module InstanceNormSetup
using LuxLib, LuxTestUtils, Random, Test, Zygote, Enzyme, NNlib
using LuxTestUtils: @jet, @test_gradients
using DispatchDoctor: allow_unstable

__is_training(::Val{training}) where {training} = training

function _setup_instancenorm(gen_f, aType, T, sz; affine::Bool=true)
    x = gen_f(T, sz) |> aType
    scale = affine ? aType(gen_f(T, sz[end - 1])) : nothing
    bias = affine ? aType(gen_f(T, sz[end - 1])) : nothing
    return x, scale, bias
end

anonact = x -> x^3

function run_instancenorm_testing(gen_f, T, sz, training, act, aType, mode, on_gpu)
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

    __f = (args...) -> sum(first(instancenorm(args..., training, act, epsilon)))
    allow_unstable() do
        @eval @test_gradients $__f $x $scale $bias gpu_testing=$on_gpu atol=$atol rtol=$rtol soft_fail=$fp16 skip_finite_differences=true
    end

    __f = (x, scale, bias) -> sum(first(instancenorm(
        x, scale, bias, training, act, epsilon)))
    if !on_gpu && !fp16 && __is_training(training)
        ∂x, ∂scale, ∂bias = Zygote.gradient(__f, x, scale, bias)

        ∂x_enz = Enzyme.make_zero(x)
        ∂scale_enz = Enzyme.make_zero(scale)
        ∂bias_enz = Enzyme.make_zero(bias)
        Enzyme.autodiff(Reverse, __f, Active, Duplicated(x, ∂x_enz),
            Duplicated(scale, ∂scale_enz), Duplicated(bias, ∂bias_enz))

        @test ∂x≈∂x_enz rtol=rtol atol=atol
        @test ∂scale≈∂scale_enz rtol=rtol atol=atol
        @test ∂bias≈∂bias_enz rtol=rtol atol=atol
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
    @testset "$mode" for (mode, aType, on_gpu) in MODES
        @testset "eltype $T, size $sz, $training $act" for (T, sz, training, act) in TEST_BLOCKS[1]
            run_instancenorm_testing(
                __generate_fixed_array, T, sz, training, act, aType, mode, on_gpu)
        end
    end
end

@testitem "Instance Norm: Group 2" tags=[:instance_norm] setup=[
    SharedTestSetup, InstanceNormSetup] begin
    @testset "$mode" for (mode, aType, on_gpu) in MODES
        @testset "eltype $T, size $sz, $training $act" for (T, sz, training, act) in TEST_BLOCKS[2]
            run_instancenorm_testing(
                __generate_fixed_array, T, sz, training, act, aType, mode, on_gpu)
        end
    end
end

@testitem "Instance Norm: Group 3" tags=[:instance_norm] setup=[
    SharedTestSetup, InstanceNormSetup] begin
    @testset "$mode" for (mode, aType, on_gpu) in MODES
        @testset "eltype $T, size $sz, $training $act" for (T, sz, training, act) in TEST_BLOCKS[3]
            run_instancenorm_testing(
                __generate_fixed_array, T, sz, training, act, aType, mode, on_gpu)
        end
    end
end

@testitem "Instance Norm: Group 4" tags=[:instance_norm] setup=[
    SharedTestSetup, InstanceNormSetup] begin
    @testset "$mode" for (mode, aType, on_gpu) in MODES
        @testset "eltype $T, size $sz, $training $act" for (T, sz, training, act) in TEST_BLOCKS[4]
            run_instancenorm_testing(
                __generate_fixed_array, T, sz, training, act, aType, mode, on_gpu)
        end
    end
end

@testitem "Instance Norm: Group 5" tags=[:instance_norm] setup=[
    SharedTestSetup, InstanceNormSetup] begin
    @testset "$mode" for (mode, aType, on_gpu) in MODES
        @testset "eltype $T, size $sz, $training $act" for (T, sz, training, act) in TEST_BLOCKS[5]
            run_instancenorm_testing(
                __generate_fixed_array, T, sz, training, act, aType, mode, on_gpu)
        end
    end
end
