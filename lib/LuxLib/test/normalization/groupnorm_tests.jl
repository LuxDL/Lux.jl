@testsetup module GroupNormSetup
using LuxLib, LuxTestUtils, Random, Test, Zygote, NNlib, Static, StableRNGs
using LuxTestUtils: check_approx

function setup_groupnorm(rng, aType, T, sz, affine)
    x = aType(randn(rng, T, sz))
    if affine
        scale = aType(randn(rng, T, sz[end - 1]))
        bias = aType(randn(rng, T, sz[end - 1]))
        return x, scale, bias
    end
    return x, nothing, nothing
end

# Bypassing all optimizations
function groupnorm_fallback(
    x::AbstractArray{<:Real,N},
    scale::LuxLib.Optional{<:AbstractVector},
    bias::LuxLib.Optional{<:AbstractVector},
    groups::Int,
    σ::F=identity,
    epsilon=1.0f-5,
) where {F,N}
    sz = size(x)
    x_reshaped = reshape(x, sz[1:(N - 2)]..., sz[N - 1] ÷ groups, groups, sz[N])
    y, _, _ = LuxLib.Impl.normalization(
        x_reshaped,
        nothing,
        nothing,
        scale,
        bias,
        LuxLib.Impl.groupnorm_reduce_dims(x),
        False(),
        nothing,
        epsilon,
        σ,
    )
    return reshape(y, sz)
end

anonact = x -> x^3

is_training(::Val{training}) where {training} = training

sumabs2groupnorm(args...) = sum(abs2, groupnorm(args...))

function run_groupnorm_testing(T, sz, groups, affine, act, aType, mode, ongpu)
    _f = (args...) -> groupnorm(args..., groups, act, epsilon)
    _f2 = (args...) -> groupnorm_fallback(args..., groups, act, epsilon)

    epsilon = LuxLib.Utils.default_epsilon(T)
    x, scale, bias = setup_groupnorm(StableRNG(0), aType, T, sz, affine)

    y = _f(x, scale, bias)
    y_simple = _f2(x, scale, bias)

    atol = 1.0f-3
    rtol = 1.0f-3

    @test y ≈ y_simple atol = atol rtol = rtol

    # Check the rrules
    ∂x, ∂scale, ∂bias = Zygote.gradient(sum ∘ _f, x, scale, bias)
    ∂x_simple, ∂scale_simple, ∂bias_simple = Zygote.gradient(sum ∘ _f2, x, scale, bias)
    if length(sz) == 5 && !ongpu
        @test_softfail check_approx(∂x, ∂x_simple; atol, rtol)
    else
        @test ∂x ≈ ∂x_simple atol = atol rtol = rtol
    end
    if affine
        @test ∂scale ≈ ∂scale_simple atol = atol rtol = rtol
        @test ∂bias ≈ ∂bias_simple atol = atol rtol = rtol
    end

    @test @inferred(groupnorm(x, scale, bias, groups, act, epsilon)) isa Any
    @jet groupnorm(x, scale, bias, groups, act, epsilon)

    @test y isa aType{T,length(sz)}
    @test size(y) == sz

    @test_gradients(
        sumabs2groupnorm,
        x,
        scale,
        bias,
        groups,
        act,
        epsilon;
        atol,
        rtol,
        soft_fail=[AutoFiniteDiff()]
    )
end

const ALL_TEST_CONFIGS = Iterators.product(
    [Float32, Float64],
    (
        (6, 2),
        # (4, 6, 2),
        # (8, 8, 8, 6, 2),
        # (3, 16, 16, 12, 2),
        (4, 4, 6, 2),
        (2, 2, 6, 2),
        # (3, 3, 12, 4),
    ),
    (2, 3),
    (true, false),
    (identity, sigmoid_fast, anonact),
)

const TEST_BLOCKS = collect(
    Iterators.partition(ALL_TEST_CONFIGS, ceil(Int, length(ALL_TEST_CONFIGS) / 2))
)

export setup_groupnorm, ALL_TEST_CONFIGS, TEST_BLOCKS, run_groupnorm_testing

end

@testitem "Group Norm: Group 1" tags = [:normalization] setup = [
    SharedTestSetup, GroupNormSetup
] begin
    @testset "$mode" for (mode, aType, ongpu, fp64) in MODES
        @testset "eltype $T, size $sz, $groups, $affine, $act" for (
            T, sz, groups, affine, act
        ) in TEST_BLOCKS[1]
            !fp64 && T == Float64 && continue
            run_groupnorm_testing(T, sz, groups, affine, act, aType, mode, ongpu)
        end
    end
end

@testitem "Group Norm: Group 2" tags = [:normalization] setup = [
    SharedTestSetup, GroupNormSetup
] begin
    @testset "$mode" for (mode, aType, ongpu, fp64) in MODES
        @testset "eltype $T, size $sz, $groups, $affine, $act" for (
            T, sz, groups, affine, act
        ) in TEST_BLOCKS[2]
            !fp64 && T == Float64 && continue
            run_groupnorm_testing(T, sz, groups, affine, act, aType, mode, ongpu)
        end
    end
end
