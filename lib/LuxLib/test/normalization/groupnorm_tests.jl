@testsetup module GroupNormSetup
using LuxLib, LuxTestUtils, Random, Test, Zygote, Enzyme, NNlib
using LuxTestUtils: @jet, @test_gradients
using DispatchDoctor: allow_unstable

function _setup_groupnorm(gen_f, aType, T, sz)
    x = gen_f(T, sz) |> aType
    scale = gen_f(T, sz[end - 1]) |> aType
    bias = gen_f(T, sz[end - 1]) |> aType
    return x, scale, bias
end

# Bypassing all optimizations
function __groupnorm_basic(
        x::AbstractArray{<:Real, N}, scale::LuxLib.Optional{<:AbstractVector},
        bias::LuxLib.Optional{<:AbstractVector}, groups::Int,
        σ::F=identity, epsilon::Real=1.0f-5) where {F, N}
    sz = size(x)
    x_reshaped = reshape(x, sz[1:(N - 2)]..., sz[N - 1] ÷ groups, groups, sz[N])
    x_ = LuxLib._normalization(x_reshaped, nothing, nothing, scale, bias,
        LuxLib._get_groupnorm_reduce_dims(x), Val(false), nothing, epsilon, σ)[1]
    return reshape(x_, sz)
end

anonact = x -> x^3

__istraining(::Val{training}) where {training} = training

function run_groupnorm_testing(gen_f, T, sz, groups, act, aType, mode, on_gpu)
    _f = (args...) -> groupnorm(args..., groups, act, epsilon)
    _f2 = (args...) -> groupnorm(args..., groups, act, epsilon)

    epsilon = LuxLib.__default_epsilon(T)
    x, scale, bias = _setup_groupnorm(gen_f, aType, T, sz)
    y = _f(x, scale, bias)

    y_simple = _f2(x, scale, bias)

    fp16 = T == Float16
    atol = fp16 ? 1.0f-2 : 1.0f-3
    rtol = fp16 ? 1.0f-2 : 1.0f-3

    @test y≈y_simple atol=atol rtol=rtol

    # Check the rrules
    if !fp16
        ∂x, ∂scale, ∂bias = Zygote.gradient(sum ∘ _f, x, scale, bias)
        ∂x_simple, ∂scale_simple, ∂bias_simple = Zygote.gradient(sum ∘ _f2, x, scale, bias)
        @test ∂x≈∂x_simple atol=atol rtol=rtol
        @test ∂scale≈∂scale_simple atol=atol rtol=rtol
        @test ∂bias≈∂bias_simple atol=atol rtol=rtol
    end

    @test @inferred(groupnorm(x, scale, bias, groups, act, epsilon)) isa Any
    @jet groupnorm(x, scale, bias, groups, act, epsilon)

    if anonact !== act
        lfn = (x, sc, b, g, act, ϵ) -> sum(groupnorm(x, sc, b, g, act, ϵ))
        @test @inferred(Zygote.gradient(lfn, x, scale, bias, groups, act, epsilon)) isa Any
    end

    @test y isa aType{T, length(sz)}
    @test size(y) == sz

    __f = (args...) -> sum(groupnorm(args..., groups, act, epsilon))
    skip_fd = act === relu
    allow_unstable() do
        @eval @test_gradients $__f $x $scale $bias gpu_testing=$on_gpu atol=$atol rtol=$rtol soft_fail=$fp16 skip_finite_differences=$(skip_fd)
    end

    __f = (x, scale, bias) -> sum(groupnorm(x, scale, bias, groups, act, epsilon))
    if !on_gpu && !fp16
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

const ALL_TEST_CONFIGS = Iterators.product([Float16, Float32, Float64],
    ((6, 2), (4, 6, 2), (8, 8, 8, 6, 2), (3, 16, 16, 12, 2),
        (4, 4, 6, 2), (2, 2, 6, 2), (3, 3, 12, 4)),
    (2, 3),
    (identity, relu, tanh_fast, sigmoid_fast, anonact))

const TEST_BLOCKS = collect(Iterators.partition(
    ALL_TEST_CONFIGS, ceil(Int, length(ALL_TEST_CONFIGS) / 5)))

export _setup_groupnorm, ALL_TEST_CONFIGS, TEST_BLOCKS, run_groupnorm_testing

end

@testitem "Group Norm: Group 1" tags=[:normalization] setup=[
    SharedTestSetup, GroupNormSetup] begin
    @testset "$mode" for (mode, aType, on_gpu) in MODES
        @testset "eltype $T, size $sz, $groups $act" for (T, sz, groups, act) in TEST_BLOCKS[1]
            run_groupnorm_testing(
                __generate_fixed_array, T, sz, groups, act, aType, mode, on_gpu)
        end
    end
end

@testitem "Group Norm: Group 2" tags=[:normalization] setup=[
    SharedTestSetup, GroupNormSetup] begin
    @testset "$mode" for (mode, aType, on_gpu) in MODES
        @testset "eltype $T, size $sz, $groups $act" for (T, sz, groups, act) in TEST_BLOCKS[2]
            run_groupnorm_testing(
                __generate_fixed_array, T, sz, groups, act, aType, mode, on_gpu)
        end
    end
end

@testitem "Group Norm: Group 3" tags=[:normalization] setup=[
    SharedTestSetup, GroupNormSetup] begin
    @testset "$mode" for (mode, aType, on_gpu) in MODES
        @testset "eltype $T, size $sz, $groups $act" for (T, sz, groups, act) in TEST_BLOCKS[3]
            run_groupnorm_testing(
                __generate_fixed_array, T, sz, groups, act, aType, mode, on_gpu)
        end
    end
end

@testitem "Group Norm: Group 4" tags=[:normalization] setup=[
    SharedTestSetup, GroupNormSetup] begin
    @testset "$mode" for (mode, aType, on_gpu) in MODES
        @testset "eltype $T, size $sz, $groups $act" for (T, sz, groups, act) in TEST_BLOCKS[4]
            run_groupnorm_testing(
                __generate_fixed_array, T, sz, groups, act, aType, mode, on_gpu)
        end
    end
end

@testitem "Group Norm: Group 5" tags=[:normalization] setup=[
    SharedTestSetup, GroupNormSetup] begin
    @testset "$mode" for (mode, aType, on_gpu) in MODES
        @testset "eltype $T, size $sz, $groups $act" for (T, sz, groups, act) in TEST_BLOCKS[5]
            run_groupnorm_testing(
                __generate_fixed_array, T, sz, groups, act, aType, mode, on_gpu)
        end
    end
end
