@testsetup module GroupNormSetup
using LuxLib

@inline __generate_fixed_array(::Type{T}, sz...) where {T} = __generate_fixed_array(T, sz)
@inline function __generate_fixed_array(::Type{T}, sz) where {T}
    return reshape(T.(collect(1:prod(sz)) ./ prod(sz)), sz...)
end
@inline __generate_fixed_array(::Type{T}, sz::Int) where {T} = T.(collect(1:sz) ./ sz)

function _setup_groupnorm(aType, T, sz, groups)
    x = __generate_fixed_array(T, sz) |> aType
    scale = __generate_fixed_array(T, sz[end - 1]) |> aType
    bias = __generate_fixed_array(T, sz[end - 1]) |> aType
    return x, scale, bias
end

function _groupnorm_generic_fallback(x, scale, bias, epsilon, groups, act)
    sz = size(x)
    N = ndims(x)
    x_reshaped = reshape(x, sz[1:(N - 2)]..., sz[N - 1] ÷ groups, groups, sz[N])
    x_, xmean, xvar = LuxLib._normalization(x_reshaped, nothing, nothing, scale, bias,
        Val(Tuple(collect(1:(N - 1)))), Val(false), nothing, epsilon, act)

    return reshape(x_, sz)
end

export _setup_groupnorm, _groupnorm_generic_fallback
end

@testitem "Group Normalization KernelAbstractions" tags=[:nworkers, :normalization] setup=[
    SharedTestSetup, GroupNormSetup] begin
    @testset "$mode" for (mode, aType, on_gpu) in MODES
        @testset "eltype $T, size $sz, ngroups $groups, $act" for T in (Float32, Float64),
            sz in ((4, 4, 6, 2), (2, 2, 6, 2), (3, 3, 12, 4)),
            groups in (2, 3),
            act in (identity, relu, tanh_fast, sigmoid_fast, x -> gelu(x))

            _f = (args...) -> groupnorm(args..., act; groups, epsilon)

            epsilon = T(1e-5)
            x, scale, bias = _setup_groupnorm(aType, T, sz, groups)

            y = _f(x, scale, bias)

            gs_x, gs_scale, gs_bias = Zygote.gradient(sum ∘ _f, x, scale, bias)

            @inferred groupnorm(x, scale, bias, act; groups, epsilon)

            # Stresses CI too much
            T !== Float16 && @jet groupnorm(x, scale, bias, act; groups, epsilon)

            @test y isa aType{T, length(sz)}
            @test size(y) == sz

            # Use the generic implementation to compare against
            __f = (args...) -> _groupnorm_generic_fallback(args..., epsilon, groups, act)

            y_ = __f(x, scale, bias)

            gs_x_, gs_scale_, gs_bias_ = Zygote.gradient(sum ∘ __f, x, scale, bias)

            # The KA implementation reorders operations manually for maximal
            # performance. Hence equality cannot be guaranteed.
            @test check_approx(y, y_; atol=1.0f-1, rtol=1.0f-1)
            @test check_approx(gs_x, gs_x_; atol=1.0f-1, rtol=1.0f-1)
            @test check_approx(gs_scale, gs_scale_; atol=1.0f-1, rtol=1.0f-1)
            @test check_approx(gs_bias, gs_bias_; atol=1.0f-1, rtol=1.0f-1)

            fp16 = T == Float16
            __f = (args...) -> sum(groupnorm(x, args..., act; groups, epsilon))
            skip_fd = act === relu
            @eval @test_gradients $__f $scale $bias gpu_testing=$on_gpu atol=1.0f-2 rtol=1.0f-2 soft_fail=$fp16 skip_finite_differences=$(skip_fd)
        end
    end
end

@testitem "Group Normalization Generic Fallback" tags=[:singleworker, :normalization] setup=[
    SharedTestSetup, GroupNormSetup] begin
    @testset "$mode" for (mode, aType, on_gpu) in MODES
        @testset "eltype $T, size $sz, ngroups $groups, $act" for T in (
                Float16, Float32, Float64),
            sz in ((4, 6, 2), (8, 8, 8, 6, 2), (3, 16, 16, 12, 2)),
            groups in (2, 3),
            act in (identity, relu, tanh_fast, sigmoid_fast, x -> x^3)

            _f = (args...) -> groupnorm(args..., act; groups, epsilon)

            epsilon = T(1e-5)
            x, scale, bias = _setup_groupnorm(aType, T, sz, groups)
            y = _f(x, scale, bias)

            @inferred groupnorm(x, scale, bias, act; groups, epsilon)

            # Stresses CI too much
            T !== Float16 && @jet groupnorm(x, scale, bias, act; groups, epsilon)

            @test y isa aType{T, length(sz)}
            @test size(y) == sz

            fp16 = T == Float16
            __f = (args...) -> sum(groupnorm(x, args..., act; groups, epsilon))
            skip_fd = act === relu
            @eval @test_gradients $__f $scale $bias gpu_testing=$on_gpu atol=1.0f-2 rtol=1.0f-2 soft_fail=$fp16 skip_finite_differences=$(skip_fd)
        end
    end
end
