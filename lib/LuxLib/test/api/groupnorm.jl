using LuxCUDA, Test
using LuxLib

include("../test_utils.jl")

function _setup_groupnorm(aType, T, sz, groups; track_stats::Bool)
    x = randn(T, sz) |> aType
    scale = randn(T, sz[end - 1]) |> aType
    bias = randn(T, sz[end - 1]) |> aType

    if track_stats
        running_mean = randn(T, groups) |> aType
        running_var = abs2.(randn(T, groups)) |> aType
        return x, scale, bias, running_mean, running_var
    else
        return x, scale, bias
    end
end

function _groupnorm_generic_fallback(x, scale, bias, running_mean, running_var, training,
                                     momentum, epsilon, groups)
    sz = size(x)
    N = ndims(x)
    x_reshaped = reshape(x, sz[1:(N - 2)]..., sz[N - 1] ÷ groups, groups, sz[N])
    x_, xmean, xvar = LuxLib._normalization(x_reshaped, running_mean, running_var, scale,
                                            bias, Val(Tuple(collect(1:(N - 1)))), training,
                                            momentum, epsilon)

    return reshape(x_, sz)
end

@testset "GroupNorm KernelAbstractions" begin for (mode, aType, on_gpu) in MODES
    for T in (Float32, Float64),
        sz in ((16, 16, 6, 4), (32, 32, 6, 4), (64, 64, 12, 4)),
        groups in (2, 3)

        _f = (args...) -> groupnorm(args...; groups, epsilon)

        epsilon = T(1e-5)
        x, scale, bias = _setup_groupnorm(aType, T, sz, groups; track_stats=false)

        y = _f(x, scale, bias)

        gs_x, gs_scale, gs_bias = Zygote.gradient((args...) -> sum(_f(args...)), x, scale,
                                                  bias)

        @inferred groupnorm(x, scale, bias; groups, epsilon)
        @jet _f(x, scale, bias) opt_broken=true
        @test y isa aType{T, 4}
        @test size(y) == sz

        # Use the generic implementation to compare against
        __f = (args...) -> _groupnorm_generic_fallback(args..., nothing, nothing, Val(true),
                                                       T(0.9), epsilon, groups)

        y_ = __f(x, scale, bias)

        gs_x_, gs_scale_, gs_bias_ = Zygote.gradient((args...) -> sum(__f(args...)), x,
                                                     scale, bias)

        # The KA implementation reorders operations manually for maximal
        # performance. Hence equality cannot be guaranteed.
        @test check_approx(y, y_; atol=1.0f-3, rtol=1.0f-3)
        @test check_approx(gs_x, gs_x_; atol=1.0f-3, rtol=1.0f-3)
        @test check_approx(gs_scale, gs_scale_; atol=1.0f-3, rtol=1.0f-3)
        @test check_approx(gs_bias, gs_bias_; atol=1.0f-3, rtol=1.0f-3)

        fp16 = T == Float16
        __f = sum ∘ _f
        @eval @test_gradients $__f $x $scale $bias gpu_testing=$on_gpu atol=1.0f-3 rtol=1.0f-3 soft_fail=$fp16
    end
end end

@testset "GroupNorm Generic Fallback" begin for (mode, aType, on_gpu) in MODES
    for T in (Float16, Float32, Float64),
        sz in ((4, 4, 6, 2), (8, 8, 6, 2), (16, 16, 12, 2)),
        groups in (2, 3),
        training in (Val(true), Val(false))

        _f = (args...) -> groupnorm(args...; groups, epsilon, training, momentum=T(0.9))

        epsilon = T(1e-5)
        x, scale, bias, rm, rv = _setup_groupnorm(aType, T, sz, groups; track_stats=true)
        y, nt = _f(x, scale, bias, rm, rv)

        @inferred groupnorm(x, scale, bias, rm, rv; groups, epsilon, training,
                            momentum=T(0.9))
        @jet _f(x, scale, bias, rm, rv) opt_broken=true

        @test y isa aType{T, 4}
        @test size(y) == sz
        @test size(nt.running_mean) == (groups,)
        @test size(nt.running_var) == (groups,)

        fp16 = T == Float16
        __f = (args...) -> sum(first(groupnorm(args..., rm, rv; groups, epsilon, training,
                                               momentum=T(0.9))))
        @eval @test_gradients $__f $x $scale $bias gpu_testing=$on_gpu atol=1.0f-2 rtol=1.0f-2 soft_fail=$fp16
    end
end end
