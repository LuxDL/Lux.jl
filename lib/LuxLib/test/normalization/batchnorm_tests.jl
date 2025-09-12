@testsetup module BatchNormSetup
using LuxLib, LuxTestUtils, Random, Test, Zygote, NNlib, Static

function setup_batchnorm(gen_f, aType, T, sz; affine::Bool=true, track_stats::Bool)
    x = aType(gen_f(T, sz))
    scale = affine ? aType(gen_f(T, sz[end - 1])) : nothing
    bias = affine ? aType(gen_f(T, sz[end - 1])) : nothing

    if track_stats
        running_mean = aType(gen_f(T, sz[end - 1]))
        running_var = aType(abs2.(gen_f(T, sz[end - 1])))
        return x, scale, bias, running_mean, running_var
    else
        return x, scale, bias, nothing, nothing
    end
end

# Bypassing all optimizations
function batchnorm_fallback(
    x::AbstractArray{<:Real,N},
    scale::LuxLib.Optional{<:AbstractVector},
    bias::LuxLib.Optional{<:AbstractVector},
    running_mean::LuxLib.Optional{<:AbstractVector},
    running_var::LuxLib.Optional{<:AbstractVector},
    training::Val,
    σ::F=identity,
    momentum=0.1f0,
    epsilon=1.0f-5,
) where {F,N}
    y, xm, xv = LuxLib.Impl.normalization(
        x,
        LuxLib.Utils.remove_tracking(running_mean),
        LuxLib.Utils.remove_tracking(running_var),
        scale,
        bias,
        LuxLib.Impl.batchnorm_reduce_dims(x),
        static(training),
        momentum,
        epsilon,
        σ,
    )
    return (
        y,
        (;
            running_mean=LuxLib.Utils.remove_tracking(LuxLib.Utils.safe_vec(xm)),
            running_var=LuxLib.Utils.remove_tracking(LuxLib.Utils.safe_vec(xv)),
        ),
    )
end

anonact = x -> x^3

is_training(::Val{training}) where {training} = training

sumabs2first(f::F, args...) where {F} = sum(abs2, first(f(args...)))

function run_batchnorm_testing(gen_f, T, sz, training, affine, track_stats, act, aType)
    epsilon = eps(T)^(5//7)
    x, scale, bias, rm, rv = setup_batchnorm(gen_f, aType, T, sz; track_stats, affine)

    y, nt = batchnorm(x, scale, bias, rm, rv, training, act, T(0.9), epsilon)
    y_simple, nt_simple = batchnorm_fallback(
        x, scale, bias, rm, rv, training, act, T(0.9), epsilon
    )

    atol = 1.0f-3
    rtol = 1.0f-3

    @test y ≈ y_simple atol = atol rtol = rtol
    if track_stats
        @test nt.running_mean ≈ nt_simple.running_mean atol = atol rtol = rtol
        @test nt.running_var ≈ nt_simple.running_var atol = atol rtol = rtol
    end

    # Check the rrules
    if is_training(training)
        _f =
            (args...) ->
                sum(first(batchnorm(args..., rm, rv, training, act, T(0.9), epsilon)))
        _f2 =
            (args...) -> sum(
                first(batchnorm_fallback(args..., rm, rv, training, act, T(0.9), epsilon)),
            )

        ∂x, ∂scale, ∂bias = Zygote.gradient(sum ∘ _f, x, scale, bias)
        ∂x_simple, ∂scale_simple, ∂bias_simple = Zygote.gradient(sum ∘ _f2, x, scale, bias)
        @test ∂x ≈ ∂x_simple atol = atol rtol = rtol
        if affine
            @test ∂scale ≈ ∂scale_simple atol = atol rtol = rtol
            @test ∂bias ≈ ∂bias_simple atol = atol rtol = rtol
        end
    end

    @test @inferred(batchnorm(x, scale, bias, rm, rv, training, act, T(0.9), epsilon)) isa
        Any
    @jet batchnorm(x, scale, bias, rm, rv, training, act, T(0.9), epsilon)

    @test y isa aType{T,length(sz)}
    @test size(y) == sz
    if rm !== nothing
        @test size(nt.running_mean) == (size(x, length(sz) - 1),)
        @test size(nt.running_var) == (size(x, length(sz) - 1),)
    end

    if is_training(training)
        @test_gradients(
            sumabs2first,
            batchnorm,
            x,
            scale,
            bias,
            Constant(rm),
            Constant(rv),
            training,
            act,
            T(0.9),
            epsilon;
            atol,
            rtol,
            soft_fail=[AutoFiniteDiff()],
        )
    end
end

const ALL_TEST_CONFIGS = Iterators.product(
    [Float32, Float64],
    # ((4, 4, 6, 2), (8, 2), (4, 4, 4, 3, 2)),
    ((4, 4, 6, 2), (8, 2)),
    (Val(true), Val(false)),
    # (true, false),
    # (true, false),
    [(true, true), (false, false)],
    (identity, sigmoid_fast, anonact),
)

const TEST_BLOCKS = collect(
    Iterators.partition(ALL_TEST_CONFIGS, ceil(Int, length(ALL_TEST_CONFIGS) / 2))
)

export setup_batchnorm, ALL_TEST_CONFIGS, TEST_BLOCKS, run_batchnorm_testing

end

@testitem "Batch Norm: Group 1" tags = [:normalization] setup = [
    SharedTestSetup, BatchNormSetup
] begin
    @testset "$mode" for (mode, aType, ongpu, fp64) in MODES
        @testset "eltype $T, size $sz, $act $affine $track_stats" for (
            T, sz, training, (affine, track_stats), act
        ) in TEST_BLOCKS[1]
            !fp64 && T == Float64 && continue
            run_batchnorm_testing(
                generate_fixed_array, T, sz, training, affine, track_stats, act, aType
            )
        end
    end
end

@testitem "Batch Norm: Group 2" tags = [:normalization] setup = [
    SharedTestSetup, BatchNormSetup
] begin
    @testset "$mode" for (mode, aType, ongpu, fp64) in MODES
        @testset "eltype $T, size $sz, $act $affine $track_stats" for (
            T, sz, training, (affine, track_stats), act
        ) in TEST_BLOCKS[2]
            !fp64 && T == Float64 && continue
            run_batchnorm_testing(
                generate_fixed_array, T, sz, training, affine, track_stats, act, aType
            )
        end
    end
end

@testitem "Batch Norm: Mixed Precision" tags = [:normalization] setup = [SharedTestSetup] begin
    @testset "$mode" for (mode, aType, ongpu, fp64) in MODES
        x = aType(rand(Float64, 4, 4, 6, 2))
        scale = aType(rand(Float32, 6))
        bias = aType(rand(Float32, 6))
        running_mean = aType(rand(Float32, 6))
        running_var = aType(rand(Float32, 6))

        y, nt = batchnorm(
            x, scale, bias, running_mean, running_var, Val(true), identity, 0.9f0, 1.0f-5
        )
        @test y isa aType{Float64,4}
        @test nt.running_mean isa aType && length(nt.running_mean) == 6
        @test nt.running_var isa aType && length(nt.running_var) == 6

        @test_gradients(
            sumabs2first,
            batchnorm,
            x,
            scale,
            bias,
            Constant(running_mean),
            Constant(running_var),
            Val(true),
            gelu,
            0.9,
            1.0e-5;
            atol=1.0f-3,
            rtol=1.0f-3
        )
    end
end
