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

function _groupnorm_generic_fallback(x, scale, bias, epsilon, groups)
    sz = size(x)
    N = ndims(x)
    x_reshaped = reshape(x, sz[1:(N - 2)]..., sz[N - 1] ÷ groups, groups, sz[N])
    x_, xmean, xvar = LuxLib._normalization(x_reshaped, nothing, nothing, scale, bias,
        Val(Tuple(collect(1:(N - 1)))), Val(false), nothing, epsilon)

    return reshape(x_, sz)
end

export _setup_groupnorm, _groupnorm_generic_fallback
end

@testitem "Group Normalization KernelAbstractions" tags=[:nworkers] setup=[
    SharedTestSetup, GroupNormSetup] begin
    @testset "$mode" for (mode, aType, on_gpu) in MODES
        @testset "eltype $T, size $sz, ngroups $groups" for T in (Float32, Float64),
            sz in ((16, 16, 6, 4), (32, 32, 6, 4), (64, 64, 12, 4)),
            groups in (2, 3)

            _f = (args...) -> groupnorm(args...; groups, epsilon)

            epsilon = T(1e-5)
            x, scale, bias = _setup_groupnorm(aType, T, sz, groups)

            y = _f(x, scale, bias)

            gs_x, gs_scale, gs_bias = Zygote.gradient(sum ∘ _f, x, scale, bias)

            @inferred groupnorm(x, scale, bias; groups, epsilon)

            @jet _f(x, scale, bias)

            @test y isa aType{T, length(sz)}
            @test size(y) == sz

            # Use the generic implementation to compare against
            __f = (args...) -> _groupnorm_generic_fallback(args..., epsilon, groups)

            y_ = __f(x, scale, bias)

            gs_x_, gs_scale_, gs_bias_ = Zygote.gradient(sum ∘ __f, x, scale, bias)

            # The KA implementation reorders operations manually for maximal
            # performance. Hence equality cannot be guaranteed.
            @test check_approx(y, y_; atol=1.0f-1, rtol=1.0f-1)
            @test check_approx(gs_x, gs_x_; atol=1.0f-1, rtol=1.0f-1)
            @test check_approx(gs_scale, gs_scale_; atol=1.0f-1, rtol=1.0f-1)
            @test check_approx(gs_bias, gs_bias_; atol=1.0f-1, rtol=1.0f-1)

            fp16 = T == Float16
            __f = (args...) -> sum(groupnorm(x, args...; groups, epsilon))
            @eval @test_gradients $__f $scale $bias gpu_testing=$on_gpu atol=1.0f-2 rtol=1.0f-2 soft_fail=$fp16
        end
    end
end

@testitem "Group Normalization Generic Fallback" tags=[:nworkers] setup=[
    SharedTestSetup, GroupNormSetup] begin
    @testset "$mode" for (mode, aType, on_gpu) in MODES
        @testset "eltype $T, size $sz, ngroups $groups" for T in (Float16, Float32, Float64),
            sz in ((4, 6, 2), (8, 8, 8, 6, 2), (3, 16, 16, 12, 2)),
            groups in (2, 3)

            T === Float16 && mode == "AMDGPU" && continue

            _f = (args...) -> groupnorm(args...; groups, epsilon)

            epsilon = T(1e-5)
            x, scale, bias = _setup_groupnorm(aType, T, sz, groups)
            y = _f(x, scale, bias)

            @inferred groupnorm(x, scale, bias; groups, epsilon)
            @jet _f(x, scale, bias)

            @test y isa aType{T, length(sz)}
            @test size(y) == sz

            fp16 = T == Float16
            __f = (args...) -> sum(groupnorm(x, args...; groups, epsilon))
            @eval @test_gradients $__f $scale $bias gpu_testing=$on_gpu atol=1.0f-2 rtol=1.0f-2 soft_fail=$fp16
        end
    end
end
