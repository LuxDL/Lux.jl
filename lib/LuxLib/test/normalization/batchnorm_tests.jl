@testsetup module BatchNormSetup
using LuxLib, LuxTestUtils, Random, Test, Zygote, Enzyme, NNlib
using LuxTestUtils: @jet, @test_gradients
using DispatchDoctor: allow_unstable

function _setup_batchnorm(gen_f, aType, T, sz; affine::Bool=true, track_stats::Bool)
    x = gen_f(T, sz) |> aType
    scale = affine ? aType(gen_f(T, sz[end - 1])) : nothing
    bias = affine ? aType(gen_f(T, sz[end - 1])) : nothing

    if track_stats
        running_mean = gen_f(T, sz[end - 1]) |> aType
        running_var = abs2.(gen_f(T, sz[end - 1])) |> aType
        return x, scale, bias, running_mean, running_var
    else
        return x, scale, bias, nothing, nothing
    end
end

# Bypassing all optimizations
function __batchnorm_basic(
        x::AbstractArray{<:Real, N}, scale::LuxLib.Optional{<:AbstractVector},
        bias::LuxLib.Optional{<:AbstractVector},
        running_mean::LuxLib.Optional{<:AbstractVector},
        running_var::LuxLib.Optional{<:AbstractVector}, training::Val,
        σ::F=identity, momentum::Real=0.1f0, epsilon::Real=1.0f-5) where {F, N}
    x_, xm, xv = LuxLib._normalization(
        x, LuxLib.__value(running_mean), LuxLib.__value(running_var), scale, bias,
        LuxLib._get_batchnorm_reduce_dims(x), training, momentum, epsilon, σ)
    return (x_, (; running_mean=LuxLib.__value(xm), running_var=LuxLib.__value(xv)))
end

anonact = x -> x^3

__istraining(::Val{training}) where {training} = training

function run_batchnorm_testing(
        gen_f, T, sz, training, affine, track_stats, act, aType, mode, on_gpu)
    epsilon = eps(T)^(5 // 7)
    x, scale, bias, rm, rv = _setup_batchnorm(gen_f, aType, T, sz; track_stats, affine)

    y, nt = batchnorm(x, scale, bias, rm, rv, training, act, T(0.9), epsilon)
    y_simple, nt_simple = __batchnorm_basic(
        x, scale, bias, rm, rv, training, act, T(0.9), epsilon)

    fp16 = T == Float16
    atol = fp16 ? 1.0f-2 : 1.0f-3
    rtol = fp16 ? 1.0f-2 : 1.0f-3

    @test y≈y_simple atol=atol rtol=rtol
    if track_stats
        @test nt.running_mean≈nt_simple.running_mean atol=atol rtol=rtol
        @test nt.running_var≈nt_simple.running_var atol=atol rtol=rtol
    end

    # Check the rrules
    if __istraining(training)
        _f = (args...) -> sum(first(batchnorm(
            args..., rm, rv, training, act, T(0.9), epsilon)))
        _f2 = (args...) -> sum(first(__batchnorm_basic(
            args..., rm, rv, training, act, T(0.9), epsilon)))

        ∂x, ∂scale, ∂bias = Zygote.gradient(sum ∘ _f, x, scale, bias)
        ∂x_simple, ∂scale_simple, ∂bias_simple = Zygote.gradient(sum ∘ _f2, x, scale, bias)
        @test ∂x≈∂x_simple atol=atol rtol=rtol
        if affine
            @test ∂scale≈∂scale_simple atol=atol rtol=rtol
            @test ∂bias≈∂bias_simple atol=atol rtol=rtol
        end
    end

    @test @inferred(batchnorm(x, scale, bias, rm, rv, training, act, T(0.9), epsilon)) isa
          Any
    @jet batchnorm(x, scale, bias, rm, rv, training, act, T(0.9), epsilon)

    @test y isa aType{T, length(sz)}
    @test size(y) == sz
    if rm !== nothing
        @test size(nt.running_mean) == (size(x, length(sz) - 1),)
        @test size(nt.running_var) == (size(x, length(sz) - 1),)
    end

    if __istraining(training) && affine
        __f = (args...) -> sum(first(batchnorm(
            x, args..., rm, rv, training, act, T(0.9), epsilon)))
        skip_fd = act === relu
        allow_unstable() do
            @eval @test_gradients $__f $scale $bias gpu_testing=$on_gpu soft_fail=$fp16 atol=$atol rtol=$rtol skip_finite_differences=$(skip_fd)
        end
    end

    if anonact !== act
        lfn = (x, sc, b, rm, rv, tr, act, ϵ) -> sum(first(batchnorm(
            x, sc, b, rm, rv, tr, act, ϵ)))
        @test @inferred(Zygote.gradient(
            lfn, x, scale, bias, rm, rv, training, act, epsilon)) isa Any
    end

    if !on_gpu && !fp16 && __istraining(training) && affine
        __f = (args...) -> sum(first(batchnorm(
            args..., rm, rv, training, act, T(0.9), epsilon)))
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
    [Float16, Float32, Float64], ((4, 4, 6, 2), (8, 2), (4, 4, 4, 3, 2)),
    (Val(true), Val(false)), (true, false), (true, false),
    (identity, relu, tanh_fast, sigmoid_fast, anonact))

const TEST_BLOCKS = collect(Iterators.partition(
    ALL_TEST_CONFIGS, ceil(Int, length(ALL_TEST_CONFIGS) / 5)))

export _setup_batchnorm, ALL_TEST_CONFIGS, TEST_BLOCKS, run_batchnorm_testing

end

@testitem "Batch Norm: Group 1" tags=[:batch_norm] setup=[SharedTestSetup, BatchNormSetup] begin
    @testset "$mode" for (mode, aType, on_gpu) in MODES
        @testset "eltype $T, size $sz, $act $affine $track_stats" for (T, sz, training, affine, track_stats, act) in TEST_BLOCKS[1]
            run_batchnorm_testing(__generate_fixed_array, T, sz, training,
                affine, track_stats, act, aType, mode, on_gpu)
        end
    end
end

@testitem "Batch Norm: Group 2" tags=[:batch_norm] setup=[SharedTestSetup, BatchNormSetup] begin
    @testset "$mode" for (mode, aType, on_gpu) in MODES
        @testset "eltype $T, size $sz, $act $affine $track_stats" for (T, sz, training, affine, track_stats, act) in TEST_BLOCKS[2]
            run_batchnorm_testing(__generate_fixed_array, T, sz, training,
                affine, track_stats, act, aType, mode, on_gpu)
        end
    end
end

@testitem "Batch Norm: Group 3" tags=[:batch_norm] setup=[SharedTestSetup, BatchNormSetup] begin
    @testset "$mode" for (mode, aType, on_gpu) in MODES
        @testset "eltype $T, size $sz, $act $affine $track_stats" for (T, sz, training, affine, track_stats, act) in TEST_BLOCKS[3]
            run_batchnorm_testing(__generate_fixed_array, T, sz, training,
                affine, track_stats, act, aType, mode, on_gpu)
        end
    end
end

@testitem "Batch Norm: Group 4" tags=[:batch_norm] setup=[SharedTestSetup, BatchNormSetup] begin
    @testset "$mode" for (mode, aType, on_gpu) in MODES
        @testset "eltype $T, size $sz, $act $affine $track_stats" for (T, sz, training, affine, track_stats, act) in TEST_BLOCKS[4]
            run_batchnorm_testing(__generate_fixed_array, T, sz, training,
                affine, track_stats, act, aType, mode, on_gpu)
        end
    end
end

@testitem "Batch Norm: Group 5" tags=[:batch_norm] setup=[SharedTestSetup, BatchNormSetup] begin
    @testset "$mode" for (mode, aType, on_gpu) in MODES
        @testset "eltype $T, size $sz, $act $affine $track_stats" for (T, sz, training, affine, track_stats, act) in TEST_BLOCKS[5]
            run_batchnorm_testing(__generate_fixed_array, T, sz, training,
                affine, track_stats, act, aType, mode, on_gpu)
        end
    end
end

@testitem "Batch Norm: Mixed Precision" tags=[:batch_norm] setup=[SharedTestSetup] begin
    @testset "$mode" for (mode, aType, on_gpu) in MODES
        x = rand(Float64, 4, 4, 6, 2) |> aType
        scale = rand(Float32, 6) |> aType
        bias = rand(Float32, 6) |> aType
        running_mean = rand(Float32, 6) |> aType
        running_var = rand(Float32, 6) |> aType

        y, nt = batchnorm(
            x, scale, bias, running_mean, running_var, Val(true), identity, 0.9f0, 1.0f-5)
        @test y isa aType{Float64, 4}
        @test nt.running_mean isa aType && length(nt.running_mean) == 6
        @test nt.running_var isa aType && length(nt.running_var) == 6

        __f = (args...) -> sum(first(batchnorm(
            x, args..., running_mean, running_var, Val(true), identity, 0.9f0, 1.0f-5)))
        allow_unstable() do
            @eval @test_gradients $__f $scale $bias gpu_testing=$on_gpu soft_fail=true atol=1.0f-2 rtol=1.0f-2
        end
    end
end
